mouse = "GN06"
dates = ["2021-01-20_10-15-16"]

rule pipeline_entry:
	input:
		tasks	= [ f"data/input/{{mouse}}/{date}/task_data/"
				for date in dates ],
		Vc	= [ f"data/input/{{mouse}}/{date}/SVD_data/Vc.mat"
				for date in dates ],
	output:
		f"data/output/{{mouse}}/SVD/svd_data.h5"
	conda:
		"code/environment.yaml"
	script:
		"code/scripts/default_entry.py"

###   legacy tests   ###

rule test_data:
	input:
		"data/output/GN06/SVD/svd_data.h5"
	conda:
		"code/environment.yaml"
	script:
		"code/test_data.py"
