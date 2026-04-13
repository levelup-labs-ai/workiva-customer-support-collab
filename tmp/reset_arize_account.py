import json
import os
from collections import defaultdict

from arize import ArizeClient
from dotenv import load_dotenv


def load_client() -> ArizeClient:
    load_dotenv(".env")
    api_key = os.environ["ARIZE_API_KEY"]
    return ArizeClient(api_key=api_key)


def list_items(response):
    for attr in (
        "experiments",
        "prompts",
        "evaluators",
        "annotation_configs",
        "datasets",
        "projects",
        "spaces",
    ):
        if hasattr(response, attr):
            return list(getattr(response, attr))
    return []


def annotation_actual_instance(config):
    actual = getattr(config, "actual_instance", None)
    if actual is not None:
        return actual
    dumped = config.model_dump()
    return dumped.get("actual_instance", dumped)


def main() -> None:
    client = load_client()
    summary = defaultdict(list)
    errors = []

    spaces = list_items(client.spaces.list())
    for space in spaces:
        space_id = space.id
        space_name = getattr(space, "name", space_id)

        experiments = list_items(client.experiments.list(space=space_id))
        for item in experiments:
            try:
                client.experiments.delete(experiment=item.id)
                summary["experiments"].append({"space": space_name, "id": item.id, "name": item.name})
            except Exception as exc:
                errors.append(f"experiment {item.id} {item.name}: {exc}")

        prompts = list_items(client.prompts.list(space=space_id))
        for item in prompts:
            try:
                client.prompts.delete(prompt=item.id)
                summary["prompts"].append({"space": space_name, "id": item.id, "name": item.name})
            except Exception as exc:
                errors.append(f"prompt {item.id} {item.name}: {exc}")

        evaluators = list_items(client.evaluators.list(space=space_id))
        for item in evaluators:
            try:
                client.evaluators.delete(evaluator=item.id)
                summary["evaluators"].append({"space": space_name, "id": item.id, "name": item.name})
            except Exception as exc:
                errors.append(f"evaluator {item.id} {item.name}: {exc}")

        configs = list_items(client.annotation_configs.list(space=space_id))
        for config in configs:
            actual = annotation_actual_instance(config)
            config_type = getattr(actual, "type", None) or actual.get("type")
            config_id = getattr(actual, "id", None) or actual.get("id")
            config_name = getattr(actual, "name", None) or actual.get("name")
            if config_type == "freeform":
                continue
            try:
                client.annotation_configs.delete(annotation_config=config_id)
                summary["annotation_configs"].append(
                    {"space": space_name, "id": config_id, "name": config_name, "type": config_type}
                )
            except Exception as exc:
                errors.append(f"annotation_config {config_id} {config_name}: {exc}")

        datasets = list_items(client.datasets.list(space=space_id))
        for item in datasets:
            try:
                client.datasets.delete(dataset=item.id)
                summary["datasets"].append({"space": space_name, "id": item.id, "name": item.name})
            except Exception as exc:
                errors.append(f"dataset {item.id} {item.name}: {exc}")

        projects = list_items(client.projects.list(space=space_id))
        for item in projects:
            try:
                client.projects.delete(project=item.id)
                summary["projects"].append({"space": space_name, "id": item.id, "name": item.name})
            except Exception as exc:
                errors.append(f"project {item.id} {item.name}: {exc}")

    verification = []
    for space in spaces:
        space_id = space.id
        space_name = getattr(space, "name", space_id)
        configs = []
        for config in list_items(client.annotation_configs.list(space=space_id)):
            actual = annotation_actual_instance(config)
            configs.append(
                {
                    "id": getattr(actual, "id", None) or actual.get("id"),
                    "name": getattr(actual, "name", None) or actual.get("name"),
                    "type": getattr(actual, "type", None) or actual.get("type"),
                }
            )
        verification.append(
            {
                "space": space_name,
                "projects": len(list_items(client.projects.list(space=space_id))),
                "datasets": len(list_items(client.datasets.list(space=space_id))),
                "experiments": len(list_items(client.experiments.list(space=space_id))),
                "prompts": len(list_items(client.prompts.list(space=space_id))),
                "evaluators": len(list_items(client.evaluators.list(space=space_id))),
                "annotation_configs": configs,
            }
        )

    print(
        json.dumps(
            {
                "deleted": {key: len(value) for key, value in summary.items()},
                "verification": verification,
                "errors": errors,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
