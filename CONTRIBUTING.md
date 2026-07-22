# Contributing

## Prepare a change

1. Open an issue for a large change.
2. Use public or synthetic test data.
3. Do not add a private model or data set.
4. Do not add an API key, password, or access token.

## Verify a change

1. Run `npm run lint` in `frontend`.
2. Run `npm run build` in `frontend`.
3. Run `python -m compileall -q backend/app run.py` in the repository root.
4. Run the secret scanner before you commit.

## Documentation rules

Use short, direct sentences. Use one term for one item. Put one action in each instruction. Define an abbreviation before you use it.
