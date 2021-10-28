mouses		= ["GN06"]
dates		= ["2021-01-20_10-15-16"]
parcelations	= ["SVD","LocaNMF"]
filters		= [ "All" ]
sides		= ["left","right"]
modalities	= ["visual", "tactile", "vistact"]

conditions	= [ f"{side}_{modal}"
			for side in sides for modal in modalities]
condition_dicts	= { f"{side}_{modal}" : {"modality" : m, "target_side_left" : s}
			for s,side in enumerate(sides) for m,modal in enumerate(modalities)}
features	= [ "mean" ]

mouse0 = mouses[0]
parcelation0 = parcelations[0]
filter0 = filters[0]


rule pipeline_entry:
	input:
		tasks	= [ f"data/input/{{mouse}}/{date}/task_data/"
				for date in dates ],
		Vc	= [ f"data/input/{{mouse}}/{date}/SVD_data/Vc.mat"
				for date in dates ],
	output:
		touch(f"data/output/{{mouse}}/SVD/data.h5"),
		touch(f"data/output/{{mouse}}/SVD/conf.yaml"),
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
		touch( f"data/output/{{mouse}}/{{parcelation}}/data.h5"),
		touch( f"data/output/{{mouse}}/{{parcelation}}/conf.yaml"),
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
		touch( f"data/output/{{mouse}}/{{parcelation}}/{{filter}}/filtered_data.h5"),
		touch( f"data/output/{{mouse}}/{{parcelation}}/{{filter}}/conf.yaml"),
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
		touch(f"data/output/{{mouse}}/{{parcelation}}/{{filter}}/{{cond}}/conf.yaml"),
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
		touch(f"data/output/{{mouse}}/{{parcelation}}/{{filter}}/{{cond}}/{{feature}}/conf.yaml"),
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
