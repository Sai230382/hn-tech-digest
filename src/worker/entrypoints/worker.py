import asyncio

import mistralai.workflows as workflows


async def amain() -> None:
    workflow_classes = workflows.discover_all_workflows_in_package("worker.workflows")
    await workflows.run_worker(workflow_classes)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
