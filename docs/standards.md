# Documentation Standards

To ensure the project remains maintainable as it scales, we adhere to the following standards for documentation.

## When to Create Documentation
We create or update files in the `docs/` folder when:

1.  **New Core Systems**: A new major component is added (e.g., a new Data Engine, a new Model Architecture, a new family of Visualizers).
2.  **Infrastructure Changes**: Changes to how the project is run, deployed, or tracked (e.g., adding W&B, changing video backends).
3.  **Architectural Decisions**: Explaining *why* a specific approach was chosen (e.g., why we use `IterableDataset` over static files).

## When NOT to Create Documentation
We do **not** create separate documentation files for:

1.  **Implementation Details**: Small code changes, refactors, or helper functions. These should be documented via Python docstrings directly in the code.
2.  **One-off Scripts**: Inspection or utility scripts in `scripts/` are self-documenting via their code and usage.
3.  **Transient Experiments**: Failed experiments or temporary hacks do not need permanent documentation.

## Documentation Structure

*   **`README.md`**: The entry point. Contains the high-level vision, quickstart guide, and links to all specific documentation. It should be the "map" of the project.
*   **`docs/`**:
    *   `*.md`: Topic-specific deep dives. File names should be descriptive (e.g., `audio_inputs.md`, `data_engine.md`).

## Artifact Management

We maintain a curated collection of successful model outputs in the `favorites/` directory.

### Naming Convention
All favorite files must follow this format to ensure chronological sorting:
`YYYY-MM-DD_HH-MM-SS_{description}.mp4`

Example: `2026-01-23_17-30-00_latent_diffusion_first_success.mp4`

### How to Save a Favorite
Use the helper script to save the current `output_test.mp4`:

```bash
uv run python scripts/save_favorite.py "description of the run"
```

This will automatically timestamp and copy the file to the `favorites/` folder.
