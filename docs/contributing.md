# Contribution Guidelines

We welcome contributions, bug reports and fixes, documentation improvements, and enhancements and ideas, from anyone at any career stage and experience.

We ask those participating to contribute to a positive community environment, as outlined by the [Contributor Convenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). 

## Report issues or problems

Issue, bugs, and feature requests can be reported in the _Issues_ tab of the [GitHub repository](https://github.com/trchudley/glacier-strain-tools/issues). 

Please give your issue a clear title. If describing a bug, include (i) a description of how your pDEMtools is set up (e.g. pip, repository cloning) and the version being used; (ii) a concise Python snippet reproducing the problem; and (iii) and explanation of why the current behaviour is wrong, and what is expected.

## Contribute to `strain_tools`

You can work directly with the development of `strain_tools` if you have a solution for an issue or enhancement. 

Before doing so, please check that the GitHub issues page to ensure that your problem or proposed solution has not already been identified or is being worked upon.

We follow a [standard git workflow](https://www.asmeurer.com/git-workflow/) for code fixes and additions. To contribute, begin by fork the pDEMtools GitHub repository into your own GitHub space, before creating an appropriately named development branch. All code is submitted via the pull request process rather than pushed directly to the main branch. Please ensure that commit and pull request messages are descriptive. The pull request will be reviewed and, if valid and suitable, accepted.

### Developer install

`strain_tools` can be installed for development by cloning the github repository. We recommend installing dependencies via `conda`/`mamba` using the included `environment.yml` file.

```bash
git clone git@github.com:trchudley/glacier-strain-tools.git
cd pdemtools/
mamba env create --file environment.yml -n straintools_env
mamba activate pdemtostraintools_envols_env
pip install -e .
```

### Unit Testing

`strain_tools` has built-in unit tests, which can be run to test the package installation. Please ensure that you have installed the `pytest` package from `pip` or `conda`/`mamba` prior to testing. Tests can then be run by running the `pytest` command from the `pdemtools` directory.

```bash
pip install pytest
cd .../glacier-strain-tools
pytest
```

<!-- Pushing changes to the GitHub repository will initiate automated CI unit testing via Github Actions. -->