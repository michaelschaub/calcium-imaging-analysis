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


rule pipeline_entry:
	input:
		tasks	= [ f"data/input/{{mouse}}/{date}/task_data/"
				for date in dates ],
		Vc	= [ f"data/input/{{mouse}}/{date}/SVD_data/Vc.mat"
				for date in dates ],
	output:
		touch(f"data/output/{{mouse}}/SVD/data.h5"),
		config = f"data/output/{{mouse}}/SVD/conf.yaml",
	log:
		f"data/output/{{mouse}}/SVD/pipeline_entry.log"
	conda:
		"code/environment.yaml"
	script:
		"code/scripts/default_entry.py"

rule parcelation:
	input:
		f"data/output/{{mouse}}/SVD/data.h5"
	output:
		f"data/output/{{mouse}}/{{parcelation}}/data.h5",
		config = f"data/output/{{mouse}}/{{parcelation}}/conf.yaml",
	log:
		f"data/output/{{mouse}}/{{parcelation}}/parcelation.log"
	conda:
		"code/environment.yaml"
	script:
		"code/scripts/parcelation.py"

rule prefilters:
	input:
		f"data/output/{{mouse}}/{{parcelation}}/data.h5"
	output:
		f"data/output/{{mouse}}/{{parcelation}}/{{filter}}/filtered_data.h5",
		config = f"data/output/{{mouse}}/{{parcelation}}/{{filter}}/conf.yaml",
	log:
		f"data/output/{{mouse}}/{{parcelation}}/{{filter}}/prefilters.log"
	conda:
		"code/environment.yaml"
	script:
		"code/scripts/prefilter.py"

rule conditions:
	input:
		f"data/output/{{mouse}}/{{parcelation}}/{{filter}}/filtered_data.h5"
	output:
		touch(f"data/output/{{mouse}}/{{parcelation}}/{{filter}}/{{cond}}/conditional_data.h5"),
		config = touch(f"data/output/{{mouse}}/{{parcelation}}/{{filter}}/{{cond}}/conf.yaml"),
	params:
		conditons = lambda wildcards : [condition_dicts[wildcards["cond"]]]
	log:
		f"data/output/{{mouse}}/{{parcelation}}/{{filter}}/{{cond}}/conditionals.log"
	conda:
		"code/environment.yaml"
	script:
		"code/scripts/conditional.py"

rule feature_calculation:
	input:
		f"data/output/{{mouse}}/{{parcelation}}/{{filter}}/{{cond}}/conditional_data.h5"
	output:
		touch(f"data/output/{{mouse}}/{{parcelation}}/{{filter}}/{{cond}}/{{feature}}/feature_data.h5"),
		config = touch(f"data/output/{{mouse}}/{{parcelation}}/{{filter}}/{{cond}}/{{feature}}/conf.yaml"),
	log:
		f"data/output/{{mouse}}/{{parcelation}}/{{filter}}/{{cond}}/{{feature}}/feature_calculation.log"
	conda:
		"code/environment.yaml"
	script:
		"code/scripts/feature.py"

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
