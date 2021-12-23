

best_feats_path = snakemake.input["best_feats"]
with open(best_feats_path, 'rb') as f:
    best_feats = pkl.load(f)

parcellation = DecompData.load(snakemake.input["parcellation"])

feat_type

rfe_graph = construct_rfe_graph(best_feats,n_nodes = len(parcellation.spatials),feat_type=feat_type)