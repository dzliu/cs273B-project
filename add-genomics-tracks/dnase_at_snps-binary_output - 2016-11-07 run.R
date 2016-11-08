# Before running, make sure you have the package 'data.table'.

# The directory where 'dnase_at_snps-binary_output.R', the .map files, and the narrowPeak folders are.
# I assume the .narrowPeak.gz files were each extracted into their own subdirectory.
dir.home = 'C:/Users/Ben/Documents/CS 273B project'
# Common header and tail to be stripped when making names for cell type data. Everything in the name
# before the last '_' and after the first '.' will be stripped first.
# 2016-11-07 NARROWPEAK source: used https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE32970 ,
# picked 'Custom Archive,' downloaded all NARROWPEAK files, extracted.
name.header = 'wgEncodeOpenChromDnase'; name.tail = 'Pk'
setwd(dir.home)
source('dnase_at_snps-binary_output.R') # load required packages and code.
files.split = tstrsplit(dir(),'.',fixed = TRUE)
map.files = dir()[files.split[[2]] == 'map']
bed.files = dir()[files.split[[2]] == 'narrowPeak' & is.na(files.split[[3]])]
bed.names = strsplit(bed.files,'_', fixed = TRUE)
bed.names = lapply(bed.names, function(x) x[length(x)])
bed.names = tstrsplit(bed.names, '.', fixed = TRUE)[[1]]
bed.names = tstrsplit(bed.names, name.header, fixed = TRUE)[[2]]
bed.names = tstrsplit(bed.names, name.tail, fixed = TRUE)[[1]]
# only use if each narrowPeak file was extracted into its own directory
bed.files = paste(bed.files, bed.files, sep = '/')

ptm = proc.time()
#takes 1-2 min per chromsome for me
for (i in map.files) { map.to.xmap(i, bed.files, bed.names); print(proc.time() - ptm) } 
