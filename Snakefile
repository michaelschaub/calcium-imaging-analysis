configfile: "config.yaml"

ruleorder: pipeline_entry > parcelation

mouses 		= config["branch_opts"]["mouses"]
dates		= config["branch_opts"]["dates"]
parcelations	= config["branch_opts"]["parcelations"]
filters		= config["branch_opts"]["filters"]

sides		= config["conditional_opts"]["sides"]
modalities	= config["conditional_opts"]["modalities"]

conditions	= [ f"{side}_{modal}"
			for side in sides for modal in modalities]
condition_dicts	= { f"{side}_{modal}" : {"modality" : m, "target_side_left" : s}
			for s,side in enumerate(sides) for m,modal in enumerate(modalities)}

features	= config["features"]

decoders = config["decode_opts"]["decoders"]
k_folds = config["decode_opts"]["k_folds"]


rule pipeline_entry:
	input:
		tasks	= [ f"data/input/{{mouse}}/{date}/task_data/"
				for date in dates ],
		Vc	= [ f"data/input/{{mouse}}/{date}/SVD_data/Vc.mat"
				for date in dates ],
		trans_params = [ f"data/input/{{mouse}}/{date}/SVD_data/opts.mat"
				  for date in dates ],
	output:
		"data/output/{mouse}/SVD/data.h5",
		config = "data/output/{mouse}/SVD/conf.yaml",
	log:
		"data/output/{mouse}/SVD/pipeline_entry.log"
	conda:
		"code/environment.yaml"
	script:
		"code/scripts/default_entry.py"

rule parcelation:
	input:
		"data/output/{mouse}/SVD/data.h5",
		config = "data/output/{mouse}/SVD/conf.yaml",
	output:
		"data/output/{mouse}/{parcelation,.+(?!SVD)}/data.h5",
		config = "data/output/{mouse}/{parcelation}/conf.yaml",
	log:
		"data/output/{mouse}/{parcelation}/parcelation.log"
	conda:
		"code/environment.yaml"
	script:
		"code/scripts/parcelation.py"

rule prefilters:
	input:
		"data/output/{mouse}/{parcelation}/data.h5",
		config = "data/output/{mouse}/{parcelation}/conf.yaml",
	output:
		"data/output/{mouse}/{parcelation}/{filter}/filtered_data.h5",
		config = "data/output/{mouse}/{parcelation}/{filter}/conf.yaml",
	log:
		"data/output/{mouse}/{parcelation}/{filter}/prefilters.log"
	conda:
		"code/environment.yaml"
	script:
		"code/scripts/prefilter.py"

rule conditions:
	input:
		"data/output/{mouse}/{parcelation}/{filter}/filtered_data.h5",
		config = "data/output/{mouse}/{parcelation}/{filter}/conf.yaml",
	output:
		"data/output/{mouse}/{parcelation}/{filter}/{cond}/conditional_data.h5",
		config = "data/output/{mouse}/{parcelation}/{filter}/{cond}/conf.yaml",
	params:
		conditions = lambda wildcards : [condition_dicts[wildcards["cond"]]]
	log:
		"data/output/{mouse}/{parcelation}/{filter}/{cond}/conditionals.log"
	conda:
		"code/environment.yaml"
	script:
		"code/scripts/conditional.py"

rule feature_calculation:
	input:
		"data/output/{mouse}/{parcelation}/{filter}/{cond}/conditional_data.h5",
		config = "data/output/{mouse}/{parcelation}/{filter}/{cond}/conf.yaml",
	output:
		"data/output/{mouse}/{parcelation}/{filter}/{cond}/{feature}/feature_data.h5",
		config = "data/output/{mouse}/{parcelation}/{filter}/{cond}/{feature}/conf.yaml",
	log:
		"data/output/{mouse}/{parcelation}/{filter}/{cond}/{feature}/feature_calculation.log"
	conda:
		"code/environment.yaml"
	script:
		"code/scripts/feature.py"

rule decoding:
	input:
		[f"data/output/{{mouse}}/{{parcelation}}/{{filter}}/{cond}/{{feature}}/feature_data.h5" for cond in conditions],
	output:
		touch("data/output/{mouse}/{parcelation}/{filter}/decoder/{feature}/{decoder}/decoder_model.pkl"),
		touch("data/output/{mouse}/{parcelation}/{filter}/decoder/{feature}/{decoder}/decoder_perf.pkl"),
		config = touch("data/output/{mouse}/{parcelation}/{filter}/decoder/{feature}/{decoder}/conf.yaml"),
	params:
		conds = conditions,
		reps = k_folds,
	log:
	   "data/output/{mouse}/{parcelation}/{filter}/decoder/{feature}/{decoder}/decoding.log",
	conda:
		"code/environment.yaml"
	script:
		"code/scripts/decoding.py"

rule plot_performance:
	input:
		[f"data/output/{{mouse}}/{{parcelation}}/{{filter}}/decoder/{{feature}}/{decoder}/decoder_perf.pkl" for decoder in decoders],
	output:
		touch("data/output/{mouse}/{parcelation}/{filter}/decoder/{feature}/plots/performance.png"),
	params:
		conds=conditions,
		decoders=decoders,
	conda:
		 "code/environment.yaml"
	script:
		  "code/scripts/plot_performance.py"

rule all_decode:
	input:
		[ f"data/output/{mouse}/{parcelation}/{filter}/decoder/{feature}/{decoder}/decoder_model.pkl"
				for mouse in mouses
				for parcelation in parcelations
				for filter in filters
				for cond in conditions
				for feature in features
		  		for decoder in decoders]

rule all_plots:
	input:
		 [ f"data/output/{mouse}/{parcelation}/{filter}/decoder/{feature}/plots/performance.png"
			   for mouse in mouses
			   for parcelation in parcelations
			   for filter in filters
			   for cond in conditions
			   for feature in features]

rule all:
	input:
		[ f"data/output/{mouse}/{parcelation}/{filter}/{cond}/{feature}/feature_data.h5"
				for mouse in mouses
				for parcelation in parcelations
				for filter in filters
				for cond in conditions
				for feature in features ]

###   legacy tests   ###

rule test_data:
	input:
		"data/output/GN06/SVD/data.h5"
	log:
		"data/output/GN06/SVD/test_data.log"
	conda:
		"code/environment.yaml"
	script:
		"code/test_data.py"

rule test_analysis:
	input:
		"data/output/GN06/SVD/data.h5"
	log:
		"data/output/GN06/SVD/test_analysis.log"
	conda:
		"code/environment.yaml"
	script:
		"code/test_analysis.py"
