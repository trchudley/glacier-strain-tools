# MkDocs Material Documentation Website

This is an example website using [MkDocs](https://www.mkdocs.org/) with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

## mkdocstrings

https://mkdocstrings.github.io/
https://mkdocstrings.github.io/python/

## Installation

Ensure the necessary dependencies are installed within your Python environment:

```bash
conda install -r mkdocs mkdocs-material mkdocs-jupyter mkdoctrings-python
```

## Running Locally

To build and serve the site locally:

```bash
mkdocs serve
```

The site will be available at `http://localhost:8000`

## Building for Production

To build the site for production:

```bash
mkdocs build
```

The output will be in the `site/` directory.

## Structure

- `mkdocs.yml` - Site configuration
- `docs/` - Documentation pages
  - `index.md` - Home page
  - `docs/pages/` - Additional pages
  - `docs/assets/` - Images and other assets

## Publishing

To deploy to GitHub Pages:

```bash
mkdocs gh-deploy
```

When you do this, a new `gh-deploy` branch should be set up (you will never need to touch this manually), and your repository should be configured correctly to initialise it at `<my-username>.github.io/<my-repository-name>`. You can check this by going to `Settings > Pages`: you should see that the site is set up to `Deploy from a branch`, and that branch is `gh-pages`. 