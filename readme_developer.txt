Directory structure:
	code: 
		matlab:
		    loop_arff (calculates folds for all arffs)
		    ilp:
			    dobscv (balanced distribution of folds in feature space)
			    ilp_slack5 (calculates folds for a single arff)
			    group_index
			    reconstruct_index
			matlab2weka:
			    loadARFF (wrapper for Weka)
		python:
		    export: (export folds in zip)
	data: input data sets
	folds: output assignments
	export: the exported arff files

Prerequisite:
	Matlab (go to "Preferences/General/Java Heap Memory" to increase the heap size if necessary)
	Weka (for loading arff files)
	Gurobi (for Integer Linear Programming. If not available, you may use Matlab's solver)
	Python 3 (for result evaluation)
	PostgreSQL (for logging)
