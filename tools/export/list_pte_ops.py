import argparse
import collections
import struct
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATED_ROOT = REPO_ROOT / "executorch" / "exir" / "_serialize" / "generated"
FLATBUFFER_ROOT = GENERATED_ROOT / "executorch_flatbuffer"

def install_schema_import_shim() -> None:
    packages = {
        "executorch": REPO_ROOT / "executorch",
        "executorch.exir": REPO_ROOT / "executorch" / "exir",
        "executorch.exir._serialize": REPO_ROOT / "executorch" / "exir" / "_serialize",
        "executorch.exir._serialize.generated": GENERATED_ROOT,
        "executorch.exir._serialize.generated.executorch_flatbuffer": FLATBUFFER_ROOT,
    }
    for name, path in packages.items():
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            sys.modules[name] = module
        module.__path__ = [str(path)]


install_schema_import_shim()
import executorch.exir._serialize.generated.executorch_flatbuffer.ExecutionPlan as execution_plan_module
import executorch.exir._serialize.generated.executorch_flatbuffer.Chain as chain_module
import executorch.exir._serialize.generated.executorch_flatbuffer.Instruction as instruction_module
import executorch.exir._serialize.generated.executorch_flatbuffer.InstructionArguments as instruction_arguments_module
import executorch.exir._serialize.generated.executorch_flatbuffer.KernelCall as kernel_call_module
import executorch.exir._serialize.generated.executorch_flatbuffer.Operator as operator_module
import executorch.exir._serialize.generated.executorch_flatbuffer.Program as program_module

program_module.ExecutionPlan = execution_plan_module.ExecutionPlan
execution_plan_module.Chain = chain_module.Chain
execution_plan_module.Operator = operator_module.Operator
chain_module.Instruction = instruction_module.Instruction
Program = program_module.Program


def read_program_bytes(path: Path) -> bytes:
    with path.open("rb") as f:
        header = f.read(64)
        if len(header) >= 40 and header[8:12] == b"eh00":
            program_size = struct.unpack_from("<Q", header, 16)[0]
            f.seek(0)
            return f.read(program_size)
        f.seek(0)
        return f.read()


def string_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def operator_name(operator) -> str:
    name = string_value(operator.Name())
    overload = string_value(operator.Overload())
    return f"{name}.{overload}" if overload else name


def plan_operator_names(plan) -> list[str]:
    return [operator_name(plan.Operators(index)) for index in range(plan.OperatorsLength())]


def kernel_call_counts(plan, names: list[str]) -> collections.Counter[str]:
    counter = collections.Counter()
    for chain_index in range(plan.ChainsLength()):
        chain = plan.Chains(chain_index)
        for instruction_index in range(chain.InstructionsLength()):
            instruction = chain.Instructions(instruction_index)
            if instruction.InstrArgsType() != instruction_arguments_module.InstructionArguments.KernelCall:
                continue
            table = instruction.InstrArgs()
            if table is None:
                continue
            kernel_call = kernel_call_module.KernelCall()
            kernel_call.Init(table.Bytes, table.Pos)
            op_index = kernel_call.OpIndex()
            name = names[op_index] if 0 <= op_index < len(names) else f"<invalid-op-index:{op_index}>"
            counter[name] += 1
    return counter


def list_ops(path: Path) -> None:
    program_bytes = read_program_bytes(path)
    program = Program.GetRootAsProgram(program_bytes, 0)
    print(f"{path}")
    print(f"  size_bytes={path.stat().st_size}")
    print(f"  program_bytes={len(program_bytes)}")
    print(f"  execution_plans={program.ExecutionPlanLength()}")

    total = collections.Counter()
    total_kernel_calls = collections.Counter()
    for plan_index in range(program.ExecutionPlanLength()):
        plan = program.ExecutionPlan(plan_index)
        plan_name = string_value(plan.Name())
        names = plan_operator_names(plan)
        counter = collections.Counter(names)
        calls = kernel_call_counts(plan, names)
        total.update(counter)
        total_kernel_calls.update(calls)
        print(
            f"  plan[{plan_index}] name={plan_name!r} "
            f"unique_ops={len(counter)} operator_table_entries={sum(counter.values())} "
            f"kernel_calls={sum(calls.values())}"
        )
        for name, count in sorted(counter.items()):
            print(f"    table={count:5d} calls={calls.get(name, 0):5d} {name}")

    if program.ExecutionPlanLength() > 1:
        print("  total")
        for name in sorted(total):
            print(f"    table={total[name]:5d} calls={total_kernel_calls.get(name, 0):5d} {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="List ExecuTorch operators embedded in .pte files.")
    parser.add_argument("pte", nargs="+", type=Path)
    args = parser.parse_args()
    for index, path in enumerate(args.pte):
        if index:
            print()
        list_ops(path)


if __name__ == "__main__":
    main()
