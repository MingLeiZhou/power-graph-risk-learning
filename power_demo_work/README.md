# DEMO package

This package was built automatically from public data sources:
- PGLib-OPF: grid case template
- OPFData: small sample of public objects from the public GCS bucket
- PowerGraph: small sample of public Figshare files

## Included files
- PGLib-OPF: `pglib/pglib_opf_case118_ieee.m` (78.45 KB)
- OPFData: `opfdata/dataset_release_1__pglib_opf_case14_ieee_0.tar.gz` (26.63 MB) -> extracted to `opfdata/dataset_release_1__pglib_opf_case14_ieee_0_extracted`
- PowerGraph: `powergraph/dataset_cascades.zip` (55.37 MB) -> extracted to `powergraph/dataset_cascades_extracted`

## Next step suggestion
Use the PGLib case as topology template, parse the OPFData files into node/edge features, and use PowerGraph as downstream benchmark/demo.