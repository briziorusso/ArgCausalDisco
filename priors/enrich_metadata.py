import argparse
import asyncio
from pathlib import Path

import pyagrum as gum
from tqdm.asyncio import tqdm

from priors.llm import GraphDescriptionBase, extract, parse_graph_description
from priors.prompt import prepare_graph_description


async def generate_graph_description(
    causal_graph: gum.BayesNet,
    model: str = "gemini-2.5-flash",
) -> GraphDescriptionBase:
    graph_prompt = prepare_graph_description(causal_graph)
    graph_description_raw = (
        (await extract(graph_prompt, None, model=model)).choices[0].message.content
    )
    assert graph_description_raw is not None, "Failed to obtain graph description"

    return await parse_graph_description(
        graph_description_raw, model=model, valid_vars=causal_graph.names()
    )


async def enrich_graph(
    causal_graph: gum.BayesNet,
    model: str = "gemini-2.5-flash",
    original_name: str | None = None,
    save_dir: str | Path | None = None,
):
    graph_description = await generate_graph_description(causal_graph, model=model)
    for name, description in graph_description.variable_descriptions.items():
        causal_graph.variableFromName(name).setDescription(description)
    causal_graph.setProperty(
        "name", original_name or graph_description.title
    )  # pyagrum ignores the name property when loading from BIFXML
    if save_dir is not None:
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{original_name or graph_description.identifier}.bifxml"
        causal_graph.saveBIFXML(str(save_path))

    return graph_description


async def process_single_file(
    bn_file: Path,
    output_dir: Path,
    model: str,
    semaphore: asyncio.Semaphore,
    rename: bool = False,
):
    """Process a single Bayesian network file with semaphore control."""
    async with semaphore:
        try:
            print(f"Processing {bn_file.name}...")

            # Load the Bayesian network
            causal_graph = gum.loadBN(str(bn_file))

            # Enrich the graph with descriptions
            graph_description = await enrich_graph(
                causal_graph,
                model=model,
                save_dir=output_dir,
                original_name=None if rename else bn_file.stem,
            )

            print(f"✓ Successfully processed {bn_file.name}")
            print(f"  Title: {graph_description.title}")
            print(f"  Variables: {len(graph_description.variable_descriptions)}")
            return True

        except Exception as e:
            print(f"✗ Error processing {bn_file.name}: {e}")
            return False


async def process_bifxml_files(
    input_dir: Path,
    output_dir: Path,
    model: str = "gemini-2.5-flash",
    max_concurrent: int = 5,
    rename: bool = False,
):
    """Process all Bayesian network files in the input directory and save enriched versions to output directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Get available file extensions from pyagrum
    available_exts = gum.availableBNExts().split("|")

    # Find all Bayesian network files in the input directory
    bn_files = []
    for ext in available_exts:
        bn_files.extend(input_dir.glob(f"*.{ext}"))

    if not bn_files:
        print(f"No Bayesian network files found in {input_dir}")
        print(f"Supported extensions: {available_exts}")
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out files that already exist in output directory
    files_to_process = []
    existing_files = []

    # Check if corresponding output file already exists
    for bn_file in bn_files:
        output_file = output_dir / f"{bn_file.stem}.bifxml"

        if output_file.exists():
            existing_files.append(bn_file.name)
        else:
            files_to_process.append(bn_file)
    print(f"Found {len(bn_files)} total Bayesian network files")
    print(
        f"Skipping {len(existing_files)} files that already exist in output directory"
    )
    print(f"Processing {len(files_to_process)} files")

    if not files_to_process:
        print("No new files to process.")
        return

    # Create semaphore to limit concurrent processing
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks for all files
    tasks = [
        asyncio.create_task(
            process_single_file(bn_file, output_dir, model, semaphore, rename=rename)
        )
        for bn_file in files_to_process
    ]

    # Wait for all tasks to complete
    success = 0
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        success += await future

    print("\nProcessing complete!")
    print(f"✓ Successfully processed: {success} files")
    print(f"✗ Failed to process: {len(tasks) - success} files")
    print(f"⏭️  Skipped (already exist): {len(existing_files)} files")
    print(f"Enriched files saved to: {output_dir}")


def main():
    """Main function to handle command-line arguments and process Bayesian network files."""
    parser = argparse.ArgumentParser(
        description="Enrich Bayesian network files with variable descriptions using LLM"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing Bayesian network files to process, supported extensions: "
        + gum.availableBNExts(),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="enriched",
        help="Output directory for enriched Bayesian network files (default: 'enriched')",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="LLM model to use for generating descriptions (default: 'gemini-2.5-flash')",
    )
    parser.add_argument(
        "-c",
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum number of files to process concurrently (default: 5)",
    )
    parser.add_argument(
        "-r",
        "--rename",
        action="store_true",
        help="Rename the output files to the LLM-generated graph title (default: False)",
    )

    args = parser.parse_args()

    asyncio.run(
        process_bifxml_files(
            args.input_dir,
            args.output_dir,
            args.model,
            args.max_concurrent,
            args.rename,
        )
    )


if __name__ == "__main__":
    main()
