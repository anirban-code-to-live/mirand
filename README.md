The *mirand* algorithm learns continuous representations for nodes in any (un)directed, (un)weighted graph. 

### Basic Usage

#### Input

- Look at the sample dataset cora. If you want to experiment on different datasets, create a folder with name of your dataset.
- Two files are required to run and generate the embedding - edgelist file for structure graph and edgelist file for content graph
- Naming convention for link structure layer: *<dataset_name>_struc.edgelist*
- Naming convention for content/attribute layer: *<dataset_name>_attr.edgelist*


#### Example
To run *mirand* on Zachary's karate club network, execute the following command from the project home directory:<br/>
	``python main.py --input-struc ../data/cora/cora_struc.edgelist --input-attr ../data/cora/cora_attr.edgelist --output ../data/cora/cora.embed --dataset=cora --dimensions=128``

#### Options
You can check out the other options available to use with *mirand* using:<br/>
	``python src/main.py --help``

#### Input
The supported input format is an edgelist:

	node1_id_int node2_id_int <weight_float, optional>
		
The graph is assumed to be undirected and unweighted by default. These options can be changed by setting the appropriate flags.

#### Output
The output file has *n+1* lines for a graph with *n* vertices. 
The first line has the following format:

	num_of_nodes dim_of_representation

The next *n* lines are as follows:
	
	node_id dim1 dim2 ... dimd

where dim1, ... , dimd is the *d*-dimensional representation learned by *mirand*.
