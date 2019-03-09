## How to run - Step by step

1. Use generate_content_edgelist.ipynb in preprocessing folder to generate the cosine_content.edgelist file. This is basically the graph generated from the content. Change the dataset name accordingly. Vary the theta parameter if required. By default, theta = 1
2. Use generate_two_level_embed_from_structure_and_content.ipynb in embedding_generation folder to generate the two level embedding. Change the dataset name accordingly. The embedding file is generated in the corresponding dataset folder in data directory. The default dimension is 128. Change the dimension parameter if you wish to generate embeddings of some other dimension.
3. The previously mentioned ipython file also converts the embedding file to a suitable csv format required for downstream data mining tasks.
4. Use the sac2vec_tasks.ipynb in tasks folder for classification, clustering tasks and visualization tasks. Change the dataset name accordingly.

## Data folder

- Look st the sample dataset cora. If you want to experiment on different datasets, create a folder with name of your dataset.
- Three files are required to run all the steps - content.csv, label.csv and reference.edgelist 
- reference.edgelist file consists of the graph structure in edgelist format.
- content.csv file is content associated with the nodes.
- label.csv contains the labels of the nodes. This file is required for classification and clustering.


## Sample command to run

python main.py --input-struc ../data/cora/cora_struc.edgelist --input-attr ../data/cora/cora_attr.edgelist --output ../data/cora/cora.embed --dataset=cora --dimensions=8