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
