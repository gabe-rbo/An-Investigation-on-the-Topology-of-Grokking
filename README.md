# An-Investigation-on-the-Topology-of-Grokking
Repository for the BRACIS submission. 

## Algorithm of Data Extraction
The algorithm for extracting the data contained in the submission was:

1. Train the Grokking Model, for up to $10^6 + 1$ epochs and the fixed _hparams_, using the data base percentage of $b \in {5, 10, 15, 20, 25, 30}$, and collect its raw activations in the folder raw-predictions- $b$ pct;
2. Apply the MP-UMAP_Reduction.py script in the raw-predictions folder with a percentil of 95, generating the respective UMAP-predictions folder;
3. Apply the MP-DE-LatentSpaceTopology.py files the past UMAP-predictions folder, with the accuracy .csv file inside of it from the respective raw-predicitions folder;
4. Analyze the results.

## Note for Grok installation
When installing the original grokking repository from OpenAI to replicate these findings, it is important to update its setup file to this:

    setup(
    name="grok",
    packages=find_packages(),
    version="0.0.1",
    install_requires=[
        "pytorch_lightning==1.0.0",
        "blobfile",
        "numpy<2",
        "torch",
        "tqdm",
        "scipy",
        "mod",
        "matplotlib",
    ],
    )

## Zips and Umaps folders
these folders contain our results from the paper. They are enourmous, but have all results and graphs. Are zipped for obvious reasons.


~ Gabriel Ribeiro, 2026, Department of Mathematics, Federal University of Minas Gerais.
