import re
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ================================================================
# INTEL 8085 TIMING / MACHINE-CYCLE GENERATOR -
# ================================================================
# Requirements:
#     python -m pip install matplotlib
#
# Generates:
#   CLK, ALE, A15-A8, AD7-AD0, RD(bar), WR(bar), IO/M, S1, S0
#
# The timing model covers the documented 8085 instruction families
# and their standard machine-cycle/T-state counts.
#
# Important 8085 bus rules:
#   * A15-A8  = high-order address bus.
#   * AD7-AD0 = multiplexed low-order address/data bus.
#   * T1      = address on A15-A8 and AD7-AD0; ALE = 1.
#   * T2/T3   = data transfer on AD7-AD0.
#   * RD(bar)/WR(bar) are active LOW, normally LOW in T2/T3.
#   * During opcode fetch, RD(bar) is LOW only in T2/T3.
#   * For an opcode-fetch cycle, A15-A8 is valid through T3;
#               the bus is shown as X in later states.
#
# This program is an educational timing-diagram generator. It models
# the externally observable bus/control timing; it does not attempt
# to model every internal 8085 signal or undocumented opcode.
# ================================================================


# --------------------------- utilities ---------------------------

REGS = {"B", "C", "D", "E", "H", "L", "M", "A"}
REG_CODE = {"B": 0, "C": 1, "D": 2, "E": 3, "H": 4, "L": 5, "M": 6, "A": 7}

COND_JUMP = {"JNZ", "JZ", "JNC", "JC", "JPO", "JPE", "JP", "JM"}
COND_CALL = {"CNZ", "CZ", "CNC", "CC", "CPO", "CPE", "CP", "CM"}
COND_RET = {"RNZ", "RZ", "RNC", "RC", "RPO", "RPE", "RP", "RM"}

PAIR_NAMES = {"B", "D", "H", "SP"}
PUSH_PAIRS = {"B", "D", "H", "PSW"}
POP_PAIRS = {"B", "D", "H", "PSW"}


def hx(value):
    """Convert 2050H, 0x2050, 2050, etc. to an integer."""
    if value is None:
        return None
    s = str(value).strip().upper().replace(" ", "")
    if not s:
        return None
    try:
        if s.endswith("H"):
            return int(s[:-1], 16)
        if s.startswith("0X"):
            return int(s[2:], 16)
        # User inputs in this program are hexadecimal by convention.
        return int(s, 16)
    except ValueError:
        return None


def fmt8(value):
    return "X" if value is None else f"{value & 0xFF:02X}H"


def fmt16(value):
    return "ADDRESS" if value is None else f"{value & 0xFFFF:04X}H"


def add16(value, amount):
    if value is None:
        return None
    return (value + amount) & 0xFFFF


def split16(value):
    if value is None:
        return "HIGH ADDRESS", "LOW ADDRESS"
    return fmt8((value >> 8) & 0xFF), fmt8(value & 0xFF)


def normalize_instruction(inst):
    s = re.sub(r"\s+", " ", inst.strip().upper())
    s = re.sub(r"\s*,\s*", ",", s)
    return s


def parse_args(inst):
    s = normalize_instruction(inst)
    if " " not in s:
        return s.split()[0], []
    op, rest = s.split(" ", 1)
    return op, [x.strip() for x in rest.split(",")]


def parse_imm8(text):
    value = hx(text)
    return fmt8(value) if value is not None else str(text).upper()


def pair_address(pair, registers):
    """Return a numeric effective address where enough register data exists."""
    pair = pair.upper()
    if pair == "B":
        b, c = registers.get("B"), registers.get("C")
        return ((b << 8) | c) if b is not None and c is not None else None
    if pair == "D":
        d, e = registers.get("D"), registers.get("E")
        return ((d << 8) | e) if d is not None and e is not None else None
    if pair == "H":
        h, l = registers.get("H"), registers.get("L")
        return ((h << 8) | l) if h is not None and l is not None else None
    if pair == "SP":
        return registers.get("SP")
    return None


# --------------------------- opcode map --------------------------

FIXED_OPCODES = {
    "NOP": "00H", "HLT": "76H",
    "RLC": "07H", "RRC": "0FH", "RAL": "17H", "RAR": "1FH",
    "DAA": "27H", "CMA": "2FH", "STC": "37H", "CMC": "3FH",
    "XCHG": "EBH", "XTHL": "E3H", "SPHL": "F9H", "PCHL": "E9H",
    "DI": "F3H", "EI": "FBH", "SIM": "30H", "RIM": "20H",
    "RET": "C9H",
    "RNZ": "C0H", "RZ": "C8H", "RNC": "D0H", "RC": "D8H",
    "RPO": "E0H", "RPE": "E8H", "RP": "F0H", "RM": "F8H",
    "PUSH B": "C5H", "PUSH D": "D5H", "PUSH H": "E5H", "PUSH PSW": "F5H",
    "POP B": "C1H", "POP D": "D1H", "POP H": "E1H", "POP PSW": "F1H",
    "STAX B": "02H", "STAX D": "12H",
    "LDAX B": "0AH", "LDAX D": "1AH",
    "INX B": "03H", "INX D": "13H", "INX H": "23H", "INX SP": "33H",
    "DCX B": "0BH", "DCX D": "1BH", "DCX H": "2BH", "DCX SP": "3BH",
    "DAD B": "09H", "DAD D": "19H", "DAD H": "29H", "DAD SP": "39H",
    "LDA": "3AH", "STA": "32H", "LHLD": "2AH", "SHLD": "22H",
    "JMP": "C3H", "JNZ": "C2H", "JZ": "CAH", "JNC": "D2H", "JC": "DAH",
    "JPO": "E2H", "JPE": "EAH", "JP": "F2H", "JM": "FAH",
    "CALL": "CDH", "CNZ": "C4H", "CZ": "CCH", "CNC": "D4H", "CC": "DCH",
    "CPO": "E4H", "CPE": "ECH", "CP": "F4H", "CM": "FCH",
    "IN": "DBH", "OUT": "D3H",
    "LXI B": "01H", "LXI D": "11H", "LXI H": "21H", "LXI SP": "31H",
    "ADI": "C6H", "ACI": "CEH", "SUI": "D6H", "SBI": "DEH",
    "ANI": "E6H", "XRI": "EEH", "ORI": "F6H", "CPI": "FEH",
}


def opcode(inst):
    s = normalize_instruction(inst)
    if s in FIXED_OPCODES:
        return FIXED_OPCODES[s]

    m = re.fullmatch(r"MOV ([BCDEHLMA]),([BCDEHLMA])", s)
    if m:
        dst, src = m.groups()
        return f"{0x40 | (REG_CODE[dst] << 3) | REG_CODE[src]:02X}H"

    m = re.fullmatch(r"MVI ([BCDEHLMA]),(.+)", s)
    if m:
        return f"{0x06 | (REG_CODE[m.group(1)] << 3):02X}H"

    for name, base in (("INR", 0x04), ("DCR", 0x05)):
        m = re.fullmatch(rf"{name} ([BCDEHLMA])", s)
        if m:
            return f"{base | (REG_CODE[m.group(1)] << 3):02X}H"

    alu_bases = {
        "ADD": 0x80, "ADC": 0x88, "SUB": 0x90, "SBB": 0x98,
        "ANA": 0xA0, "XRA": 0xA8, "ORA": 0xB0, "CMP": 0xB8
    }
    for name, base in alu_bases.items():
        m = re.fullmatch(rf"{name} ([BCDEHLMA])", s)
        if m:
            return f"{base | REG_CODE[m.group(1)]:02X}H"

    # Parameterized fixed-opcode families.
    for name in (
        "LDA", "STA", "LHLD", "SHLD", "JMP",
        "JNZ", "JZ", "JNC", "JC", "JPO", "JPE", "JP", "JM",
        "CALL", "CNZ", "CZ", "CNC", "CC", "CPO", "CPE", "CP", "CM",
        "IN", "OUT",
        "ADI", "ACI", "SUI", "SBI", "ANI", "XRI", "ORI", "CPI",
    ):
        if s.startswith(name + " "):
            return FIXED_OPCODES[name]

    # Parameterized register/pair families whose opcode is fixed by
    # the first operand.
    for family in ("LXI", "PUSH", "POP", "INX", "DCX", "DAD", "STAX", "LDAX"):
        if s.startswith(family + " "):
            parts = s.split()
            if len(parts) >= 2:
                key = f"{family} {parts[1].split(',')[0]}"
                if key in FIXED_OPCODES:
                    return FIXED_OPCODES[key]

    for name in COND_RET:
        if s == name:
            return FIXED_OPCODES[name]

    # RST 0 ... RST 7
    m = re.fullmatch(r"RST\s*([0-7])", s)
    if m:
        return f"{0xC7 + 8 * int(m.group(1)):02X}H"

    return "UNKNOWN"


# ----------------------- machine cycle class ---------------------

class Cycle:
    def __init__(
        self,
        name,
        kind,
        tstates,
        addr=None,
        addr_label=None,
        data="",
        write="",
        description="",
        bus_after_t1=True,
    ):
        self.name = name
        self.kind = kind
        self.tstates = tstates
        self.addr = addr
        self.addr_label = addr_label
        self.data = data
        self.write = write
        self.description = description
        self.bus_after_t1 = bus_after_t1


# ------------------------ cycle constructors ---------------------

def OF(name, tstates=4, pc=None, description="Opcode fetch"):
    return Cycle(name, "OF", tstates, addr=pc, description=description)


def MR(name, addr=None, addr_label=None, data="DATA", description="Memory read"):
    return Cycle(
        name, "MR", 3, addr=addr, addr_label=addr_label,
        data=data, description=description
    )


def MW(name, addr=None, addr_label=None, write="DATA", description="Memory write"):
    return Cycle(
        name, "MW", 3, addr=addr, addr_label=addr_label,
        write=write, description=description
    )


def IOR(name, port=None, data="INPUT DATA", description="I/O read"):
    return Cycle(
        name, "IOR", 3, addr=port, data=data, description=description
    )


def IOW(name, port=None, write="A", description="I/O write"):
    return Cycle(
        name, "IOW", 3, addr=port, write=write, description=description
    )


# ----------------------- instruction timing -----------------------

def make_cycles(
    inst,
    pc=0x2000,
    sp=0x4000,
    memory_data="DATA",
    accumulator="A",
    condition=True,
    registers=None,
):
    """
    Build the 8085 machine-cycle sequence.

    condition=True  -> conditional branch/call/return is taken
    condition=False -> conditional branch/call/return is not taken

    registers may contain:
        B, C, D, E, H, L, SP
    values as integers.
    """
    s = normalize_instruction(inst)
    op, args = parse_args(s)
    registers = registers or {}

    cycles = []

    # ----- MOV -----
    m = re.fullmatch(r"MOV ([BCDEHLMA]),([BCDEHLMA])", s)
    if m:
        dst, src = m.groups()
        cycles.append(OF("M1", 4, pc))

        if dst == "M":
            addr = pair_address("H", registers)
            cycles.append(
                MW("M2", addr=addr, addr_label=None if addr is not None else "HL",
                   write=src if src != "M" else "DATA",
                   description="Memory write through HL")
            )
        elif src == "M":
            addr = pair_address("H", registers)
            cycles.append(
                MR("M2", addr=addr, addr_label=None if addr is not None else "HL",
                   data=memory_data,
                   description="Memory read through HL")
            )
        return cycles

    # ----- MVI -----
    m = re.fullmatch(r"MVI ([BCDEHLMA]),(.+)", s)
    if m:
        reg, value = m.groups()
        cycles.append(OF("M1", 4, pc))
        cycles.append(
            MR("M2", addr=add16(pc, 1), data=parse_imm8(value),
               description="Immediate data fetch")
        )
        if reg == "M":
            addr = pair_address("H", registers)
            cycles.append(
                MW("M3", addr=addr,
                   addr_label=None if addr is not None else "HL",
                   write=parse_imm8(value),
                   description="Write immediate data to memory")
            )
        return cycles

    # ----- LXI -----
    if op == "LXI" and len(args) == 2:
        value = args[1]
        cycles.extend([
            OF("M1", 4, pc),
            MR("M2", addr=add16(pc, 1), data=f"{value.upper()} LOW",
               description="Low-order immediate byte"),
            MR("M3", addr=add16(pc, 2), data=f"{value.upper()} HIGH",
               description="High-order immediate byte"),
        ])
        return cycles

    # ----- direct address: LDA / STA / LHLD / SHLD -----
    if op in {"LDA", "STA", "LHLD", "SHLD"} and args:
        target = hx(args[0])
        low = fmt8(target & 0xFF) if target is not None else "LOW ADDRESS"
        high = fmt8((target >> 8) & 0xFF) if target is not None else "HIGH ADDRESS"

        cycles.extend([
            OF("M1", 4, pc),
            MR("M2", addr=add16(pc, 1), data=low,
               description="Low-order address byte"),
            MR("M3", addr=add16(pc, 2), data=high,
               description="High-order address byte"),
        ])

        if op == "LDA":
            cycles.append(
                MR("M4", addr=target, addr_label=None if target is not None else "TARGET",
                   data=memory_data,
                   description="Read data from target address")
            )
        elif op == "STA":
            cycles.append(
                MW("M4", addr=target, addr_label=None if target is not None else "TARGET",
                   write=accumulator,
                   description="Write accumulator to target address")
            )
        elif op == "LHLD":
            cycles.extend([
                MR("M4", addr=target, addr_label=None if target is not None else "TARGET",
                   data=f"{memory_data} LOW",
                   description="Read L from target address"),
                MR("M5", addr=add16(target, 1),
                   addr_label=None if target is not None else "TARGET+1",
                   data=f"{memory_data} HIGH",
                   description="Read H from target address + 1"),
            ])
        elif op == "SHLD":
            cycles.extend([
                MW("M4", addr=target, addr_label=None if target is not None else "TARGET",
                   write="L",
                   description="Write L to target address"),
                MW("M5", addr=add16(target, 1),
                   addr_label=None if target is not None else "TARGET+1",
                   write="H",
                   description="Write H to target address + 1"),
            ])
        return cycles

    # ----- JMP / conditional JMP -----
    if op in COND_JUMP | {"JMP"}:
        cycles.append(OF("M1", 4, pc))
        cycles.append(
            MR("M2", addr=add16(pc, 1), data="LOW TARGET",
               description="Low-order jump address")
        )
        if op == "JMP" or condition:
            cycles.append(
                MR("M3", addr=add16(pc, 2), data="HIGH TARGET",
                   description="High-order jump address")
            )
        return cycles

    # ----- CALL / conditional CALL -----
    if op in COND_CALL | {"CALL"}:
        if op in COND_CALL and not condition:
            return [
                OF("M1", 6, pc,
                   description="Conditional CALL: condition false; not taken"),
                MR("M2", addr=add16(pc, 1), data="LOW TARGET",
                   description="Low-order target byte")
            ]

        cycles.extend([
            OF("M1", 6, pc, description="CALL opcode fetch"),
            MR("M2", addr=add16(pc, 1), data="LOW TARGET",
               description="Low-order target byte"),
            MR("M3", addr=add16(pc, 2), data="HIGH TARGET",
               description="High-order target byte"),
            MW("M4", addr=add16(sp, -1), write="RETURN HIGH",
               description="Push return-address high byte"),
            MW("M5", addr=add16(sp, -2), write="RETURN LOW",
               description="Push return-address low byte"),
        ])
        return cycles

    # ----- RET / conditional RET -----
    if op in COND_RET | {"RET"}:
        if op in COND_RET and not condition:
            return [
                OF("M1", 6, pc,
                   description="Conditional RET: condition false; not taken")
            ]

        # Unconditional RET = 10T: M1=4T, M2=3T, M3=3T
        # Conditional RET taken = 12T: M1=6T, M2=3T, M3=3T
        first_t = 6 if op in COND_RET else 4
        cycles.extend([
            OF("M1", first_t, pc,
               description="RET opcode fetch"),
            MR("M2", addr=sp, data="LOW RETURN ADDRESS",
               description="Read low return-address byte"),
            MR("M3", addr=add16(sp, 1), data="HIGH RETURN ADDRESS",
               description="Read high return-address byte"),
        ])
        return cycles

    # ----- PUSH -----
    if op == "PUSH":
        pair = args[0] if args else "PSW"
        return [
            OF("M1", 6, pc, description="PUSH opcode fetch"),
            MW("M2", addr=add16(sp, -1), write=f"{pair} HIGH",
               description="Stack write: high byte"),
            MW("M3", addr=add16(sp, -2), write=f"{pair} LOW",
               description="Stack write: low byte"),
        ]

    # ----- POP -----
    if op == "POP":
        pair = args[0] if args else "PSW"
        return [
            OF("M1", 4, pc, description="POP opcode fetch"),
            MR("M2", addr=sp, data=f"{pair} LOW",
               description="Stack read: low byte"),
            MR("M3", addr=add16(sp, 1), data=f"{pair} HIGH",
               description="Stack read: high byte"),
        ]

    # ----- IN / OUT -----
    if op in {"IN", "OUT"} and args:
        port = hx(args[0])
        cycles = [
            OF("M1", 4, pc, description=f"{op} opcode fetch"),
            MR("M2", addr=add16(pc, 1), data=fmt8(port),
               description="Fetch 8-bit I/O port address"),
        ]
        if op == "IN":
            cycles.append(
                IOR("M3", port=port, data="INPUT DATA",
                    description="I/O read from port")
            )
        else:
            cycles.append(
                IOW("M3", port=port, write=accumulator,
                    description="I/O write to port")
            )
        return cycles

    # ----- STAX / LDAX -----
    if op in {"STAX", "LDAX"} and args:
        pair = args[0]
        addr = pair_address(pair, registers)
        if op == "STAX":
            return [
                OF("M1", 4, pc),
                MW("M2", addr=addr,
                   addr_label=None if addr is not None else pair,
                   write=accumulator,
                   description=f"Store accumulator through {pair}")
            ]
        return [
            OF("M1", 4, pc),
            MR("M2", addr=addr,
               addr_label=None if addr is not None else pair,
               data=memory_data,
               description=f"Load accumulator through {pair}")
        ]

    # ----- INR M / DCR M -----
    if op in {"INR", "DCR"} and args and args[0] == "M":
        addr = pair_address("H", registers)
        return [
            OF("M1", 4, pc),
            MR("M2", addr=addr,
               addr_label=None if addr is not None else "HL",
               data=memory_data,
               description="Read memory through HL"),
            MW("M3", addr=addr,
               addr_label=None if addr is not None else "HL",
               write="MODIFIED DATA",
               description="Write modified memory data"),
        ]

    # ----- all other 8-bit register ALU operations -----
    m = re.fullmatch(
        r"(ADD|ADC|SUB|SBB|ANA|XRA|ORA|CMP) ([BCDEHLMA])", s
    )
    if m:
        _, reg = m.groups()
        if reg == "M":
            addr = pair_address("H", registers)
            return [
                OF("M1", 4, pc),
                MR("M2", addr=addr,
                   addr_label=None if addr is not None else "HL",
                   data=memory_data,
                   description="Read operand through HL"),
            ]
        return [OF("M1", 4, pc)]

    # ----- immediate ALU -----
    if op in {"ADI", "ACI", "SUI", "SBI", "ANI", "XRI", "ORI", "CPI"}:
        value = args[0] if args else "DATA"
        return [
            OF("M1", 4, pc),
            MR("M2", addr=add16(pc, 1), data=parse_imm8(value),
               description="Fetch immediate operand"),
        ]

    # ----- INX / DCX: 6T single machine cycle -----
    if op in {"INX", "DCX"} and args and args[0] in PAIR_NAMES:
        return [
            OF("M1", 6, pc,
               description=f"{op} {args[0]}: opcode fetch + internal operation")
        ]

    # ----- DAD: 10T = 4T fetch + internal operation -----
    if op == "DAD" and args and args[0] in PAIR_NAMES:
        return [
            OF("M1", 10, pc,
               description=f"DAD {args[0]}: opcode fetch + internal operation")
        ]

    # ----- PCHL / SPHL: 6T -----
    if op in {"PCHL", "SPHL"}:
        return [
            OF("M1", 6, pc,
               description=f"{op}: opcode fetch + internal operation")
        ]

    # ----- XTHL: 16T -----
    if op == "XTHL":
        return [
            OF("M1", 4, pc),
            MR("M2", addr=sp, data="OLD L",
               description="Read stack byte into L"),
            MR("M3", addr=add16(sp, 1), data="OLD H",
               description="Read stack byte into H"),
            MW("M4", addr=sp, write="OLD L? / H DATA",
               description="Write register byte to stack"),
            MW("M5", addr=add16(sp, 1), write="H",
               description="Write register byte to stack"),
        ]

    # ----- RST 0 ... RST 7 -----
    m = re.fullmatch(r"RST\s*([0-7])", s)
    if m:
        n = int(m.group(1))
        return [
            OF("M1", 6, pc, description=f"RST {n} opcode fetch"),
            MW("M2", addr=add16(sp, -1), write="RETURN HIGH",
               description="Push return-address high byte"),
            MW("M3", addr=add16(sp, -2), write="RETURN LOW",
               description="Push return-address low byte"),
        ]

    # ----- HLT -----
    if op == "HLT":
        return [
            OF("M1", 4, pc, description="HLT opcode fetch"),
            Cycle("M2", "HALT", 1, description="HALT state begins; persists until reset/interrupt")
        ]

    # ----- one-byte instructions -----
    one_byte_4t = {
        "NOP", "RLC", "RRC", "RAL", "RAR", "DAA", "CMA", "STC", "CMC",
        "XCHG", "DI", "EI", "SIM", "RIM"
    }
    if op in one_byte_4t:
        return [OF("M1", 4, pc)]

    # Unknown instruction: do not silently pretend it is a 4T instruction.
    raise ValueError(
        f"Instruction not recognized: {s}\n"
        "Use a documented 8085 mnemonic and valid operands."
    )


# ------------------------- control signals ------------------------

def signals(kind, t):
    """
    Returns:
        ALE, RD(bar), WR(bar), IO/M, S1, S0

    For all external bus cycles:
        T1: ALE=1, RD=1, WR=1
        T2/T3:
            read  -> RD=0
            write -> WR=0
        status lines remain valid for the complete machine cycle.
    """
    if kind == "OF":
        return (1, 1, 1, 0, 1, 1) if t == 1 else (0, 0, 1, 0, 1, 1) if t in (2, 3) else (0, 1, 1, 0, 1, 1)

    if kind == "MR":
        return (1, 1, 1, 0, 1, 0) if t == 1 else (0, 0, 1, 0, 1, 0)

    if kind == "MW":
        return (1, 1, 1, 0, 0, 1) if t == 1 else (0, 1, 0, 0, 0, 1)

    if kind == "IOR":
        return (1, 1, 1, 1, 1, 0) if t == 1 else (0, 0, 1, 1, 1, 0)

    if kind == "IOW":
        return (1, 1, 1, 1, 0, 1) if t == 1 else (0, 1, 0, 1, 0, 1)

    # HALT state: no new bus cycle is initiated.
    return (0, 1, 1, 0, 0, 0)


# ----------------------- bus value generation ---------------------

def bus_values(cycle, inst):
    """
    Return a (high-bus, low/data-bus) pair for each T-state.

    T1:
        address on A15-A8 and AD7-AD0.
    T2/T3:
        A15-A8 remains valid; AD7-AD0 carries transfer data.
    Later states:
        shown as X because the external bus is not represented as
        a continuously driven address/data bus there.
    """
    opbyte = opcode(inst)
    high_addr, low_addr = split16(cycle.addr)

    # I/O timing diagrams conventionally show the 8-bit port address
    # on the address bus during T1. This is not a 16-bit memory address.
    if cycle.kind in {"IOR", "IOW"}:
        port = cycle.addr
        if port is None:
            high_addr = "PORT"
            low_addr = "PORT"
        else:
            high_addr = fmt8(port)
            low_addr = fmt8(port)

    if cycle.addr is None and cycle.addr_label:
        high_addr = f"{cycle.addr_label} HIGH"
        low_addr = f"{cycle.addr_label} LOW"

    high = []
    low = []

    for t in range(1, cycle.tstates + 1):
        if cycle.kind == "HALT":
            high.append("X")
            low.append("X")
            continue

        if t == 1:
            high.append(high_addr)
            low.append(low_addr)
        elif t in (2, 3):
            # A15-A8 remains valid through T3 for normal external cycles.
            high.append(high_addr)

            if cycle.kind == "OF":
                low.append(opbyte)
            elif cycle.kind in {"MR", "IOR"}:
                low.append(cycle.data or "DATA")
            elif cycle.kind in {"MW", "IOW"}:
                low.append(cycle.write or "DATA")
            else:
                low.append("X")
        else:
            high.append("X")
            low.append("X")

    return high, low


# ----------------------------- plotting ---------------------------

def digital(ax, values, y, label):
    h = 0.62
    x = []
    yy = []

    for i, value in enumerate(values):
        x.extend([i, i + 1])
        yy.extend([y + h * value, y + h * value])

        if i < len(values) - 1 and value != values[i + 1]:
            x.extend([i + 1, i + 1])
            yy.extend([y + h * value, y + h * values[i + 1]])

    ax.plot(x, yy, linewidth=1.5)
    ax.text(-0.25, y + h / 2, label,
            ha="right", va="center", fontsize=9, fontweight="bold")


def bus(ax, values, y, label):
    h = 0.72
    ax.text(-0.25, y + h / 2, label,
            ha="right", va="center", fontsize=9, fontweight="bold")

    for i, value in enumerate(values):
        rect = Rectangle(
            (i, y), 1, h,
            fill=False, linewidth=1.15
        )
        ax.add_patch(rect)
        if value:
            ax.text(
                i + 0.5, y + h / 2, value,
                ha="center", va="center", fontsize=8
            )


def timing_diagram(inst, cycles):
    ts_labels = []
    ale, rd, wr, iom, s1, s0 = [], [], [], [], [], []
    high_bus, low_bus = [], []
    cycle_starts = []

    total_t = 0

    for cycle in cycles:
        cycle_starts.append((total_t, cycle.name, cycle.tstates))

        high, low = bus_values(cycle, inst)

        for t in range(cycle.tstates):
            # Convert zero-based index to 1-based T number.
            t_number = t + 1
            ts_labels.append(f"T{t_number}")

            a, r, w, io_m, status1, status0 = signals(cycle.kind, t_number)
            ale.append(a)
            rd.append(r)
            wr.append(w)
            iom.append(io_m)
            s1.append(status1)
            s0.append(status0)

            high_bus.append(high[t])
            low_bus.append(low[t])
            total_t += 1

    fig, ax = plt.subplots(
        figsize=(max(15, total_t * 0.72), 12)
    )

    y = {
        "CLK": 9,
        "ALE": 8,
        "A15-A8": 7,
        "AD7-AD0": 6,
        "RD": 5,
        "WR": 4,
        "IO/M": 3,
        "S1": 2,
        "S0": 1,
    }

    # Correct 8085 clock:
    # one complete clock period = one T-state.
    # Rising edge at half T, falling edge at T boundary.
    clock_low = y["CLK"]
    clock_high = y["CLK"] + 0.62

    x = []
    yy = []

    for i in range(total_t):
        x.extend([i, i + 0.5])
        yy.extend([clock_low, clock_low])

        x.extend([i + 0.5, i + 0.5])
        yy.extend([clock_low, clock_high])

        x.extend([i + 0.5, i + 1.0])
        yy.extend([clock_high, clock_high])

        if i < total_t - 1:
            x.extend([i + 1.0, i + 1.0])
            yy.extend([clock_high, clock_low])

    ax.plot(x, yy, linewidth=1.5)
    ax.text(-0.25, y["CLK"] + 0.31, "CLK",
            ha="right", va="center", fontsize=9, fontweight="bold")

    digital(ax, ale, y["ALE"], "ALE")
    bus(ax, high_bus, y["A15-A8"], "A15-A8")
    bus(ax, low_bus, y["AD7-AD0"], "AD7-AD0")
    digital(ax, rd, y["RD"], "RD̅")
    digital(ax, wr, y["WR"], "WR̅")
    digital(ax, iom, y["IO/M"], "IO/M")
    digital(ax, s1, y["S1"], "S1")
    digital(ax, s0, y["S0"], "S0")

    for i, label in enumerate(ts_labels):
        ax.text(i + 0.5, 10.15, label,
                ha="center", va="bottom", fontsize=8)

    for start, name, width in cycle_starts:
        ax.axvline(start, linestyle="--", linewidth=0.8)
        ax.text(
            start + width / 2, 10.72, name,
            ha="center", va="bottom",
            fontsize=10, fontweight="bold"
        )

    ax.axvline(total_t, linestyle="--", linewidth=0.8)

    ax.set_xlim(-1.4, total_t)
    ax.set_ylim(0.4, 11.15)
    ax.set_yticks([])
    ax.set_xticks(range(total_t + 1))
    ax.set_xticklabels(range(total_t + 1), fontsize=7)

    ax.set_title(
        f"8085 Timing Diagram — {normalize_instruction(inst)}",
        fontsize=14, fontweight="bold", pad=28
    )

    ax.text(
        0.5, 0.015,
        "A15-A8 = high-order address | "
        "AD7-AD0 = low-order address in T1, data in T2/T3 | "
        "RD̅ and WR̅ are active LOW",
        transform=ax.transAxes,
        ha="center", fontsize=9
    )

    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.35)
    plt.tight_layout()
    return fig


# --------------------------- text summary -------------------------

def print_summary(inst, cycles, pc, sp):
    print("\n" + "=" * 78)
    print("8085 INSTRUCTION TIMING")
    print("=" * 78)
    print(f"Instruction : {normalize_instruction(inst)}")
    print(f"Opcode      : {opcode(inst)}")
    print(f"Start PC    : {fmt16(pc)}")
    print(f"Stack Pointer: {fmt16(sp)}")
    print("\nMachine cycles:")

    total = 0
    for c in cycles:
        print(
            f"  {c.name:<3} {c.kind:<4} {c.tstates:>2} T-states  "
            f"{c.description}"
        )
        total += c.tstates

    print(f"\nTotal T-states: {total}")

    print("\nControl-signal rules:")
    print("  Opcode fetch : IO/M=0, S1=1, S0=1, RD̅ LOW in T2/T3")
    print("  Memory read  : IO/M=0, S1=1, S0=0, RD̅ LOW in T2/T3")
    print("  Memory write : IO/M=0, S1=0, S0=1, WR̅ LOW in T2/T3")
    print("  I/O read     : IO/M=1, S1=1, S0=0, RD̅ LOW in T2/T3")
    print("  I/O write    : IO/M=1, S1=0, S0=1, WR̅ LOW in T2/T3")


# ------------------------------- main -----------------------------

def main():
    print("\n" + "=" * 78)
    print("     INTEL 8085 TIMING & MACHINE-CYCLE GENERATOR")
    print("=" * 78)

    print("\nExamples:")
    print("  NOP")
    print("  MOV A,B")
    print("  MOV A,M")
    print("  MOV M,A")
    print("  MVI A,32H")
    print("  MVI M,32H")
    print("  LDA 2050H")
    print("  STA 3050H")
    print("  LHLD 2050H")
    print("  SHLD 3050H")
    print("  LXI H,2050H")
    print("  JMP 2500H")
    print("  JNZ 2500H")
    print("  CALL 3000H")
    print("  CNZ 3000H")
    print("  RET")
    print("  RNZ")
    print("  PUSH B")
    print("  POP B")
    print("  IN 80H")
    print("  OUT 40H")
    print("  INX H")
    print("  DAD H")
    print("  XTHL")

    inst = input("\nEnter 8085 instruction: ").strip()
    if not inst:
        print("No instruction entered.")
        return

    raw = input("Starting program counter [2000H]: ").strip()
    pc = hx(raw) if raw else 0x2000
    if pc is None:
        print("Invalid PC. Using 2000H.")
        pc = 0x2000

    raw = input("Stack pointer [4000H]: ").strip()
    sp = hx(raw) if raw else 0x4000
    if sp is None:
        print("Invalid SP. Using 4000H.")
        sp = 0x4000

    memory_data = input("Memory data [DATA]: ").strip().upper() or "DATA"
    accumulator = input("Accumulator value for writes [A]: ").strip().upper() or "A"

    print("\nOptional register values for numeric effective addresses.")
    print("Press Enter to keep a register symbolic.")
    registers = {}

    for reg in ("B", "C", "D", "E", "H", "L"):
        raw = input(f"{reg} [symbolic]: ").strip()
        if raw:
            value = hx(raw)
            if value is None or not 0 <= value <= 0xFF:
                print(f"Invalid {reg}; keeping it symbolic.")
            else:
                registers[reg] = value

    registers["SP"] = sp

    op, _ = parse_args(inst)

    condition = True
    if op in COND_JUMP | COND_CALL | COND_RET:
        answer = input("Is the condition TRUE/taken? [Y/n]: ").strip().upper()
        condition = answer != "N"

    try:
        cycles = make_cycles(
            inst=inst,
            pc=pc,
            sp=sp,
            memory_data=memory_data,
            accumulator=accumulator,
            condition=condition,
            registers=registers,
        )
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        return

    print_summary(inst, cycles, pc, sp)

    print("\nGenerating timing diagram...")
    fig = timing_diagram(inst, cycles)
    plt.show()


if __name__ == "__main__":
    main()
