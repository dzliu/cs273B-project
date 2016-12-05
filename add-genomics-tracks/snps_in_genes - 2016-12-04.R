# read in and clean up BED files
# for whole-gene, create and clean up additional BED file with 5 KB pad on both sides
# add these records to .xmap files
# extract corresponding SNPs
# Find if any of these SNPs are not in the 'accessible' sets
require(data.table)
setwd('C:/Users/Ben/Documents/CS 273B project')

# read in both files
# All are from UCSC Genome Browser hg19 human assembly, knownGene table
# Genes_all is 'Whole Gene' download option, Genes_exons is exons only.
genes.bed = fread('Genes_all.txt')
exons.bed = fread('Genes_exons.txt')
# Only columns of interest are V1, V2, V3; rename as 'chrom','chromStart','chromEnd'; discard rest
# generate separate table for each chromosome
preproc.bed = function(bed.in) {
  bed.use = bed.in[,c('V1', 'V2', 'V3'),with=FALSE]
  setnames(bed.use, c('chrom', 'chromStart', 'chromEnd'))
  beds.out = list()
  chrom.names = unique(bed.use$chrom)
  for(i in 1:length(chrom.names)) {
    beds.out[[i]] = bed.use[bed.use$chrom == chrom.names[i],]
    names(beds.out)[i] = chrom.names[i]
  }
  return(beds.out)
}
genes.bed.proc = preproc.bed(genes.bed); exons.bed.proc = preproc.bed(exons.bed)
# 60 entries in these; mtDNA, X, Y, and some minor things, probably some kind of sequencing artifact.
# Keep only autosomal chromosomes.
aut.names = paste0('chr',1:22)
genes.bed.proc = genes.bed.proc[names(genes.bed.proc) %in% aut.names]
exons.bed.proc = exons.bed.proc[names(exons.bed.proc) %in% aut.names]

genes.bed.proc.clean = lapply(genes.bed.proc, clean.bed)
exons.bed.proc.clean = lapply(exons.bed.proc, clean.bed)

# Extend genes.bed.proc.clean genes to have 5 kb flanking regions added
genes.bed.padded = lapply(genes.bed.proc.clean,pad.genes,pad.size = 5000)

# Write these all as new .bed files

write.table(do.call(rbind, genes.bed.proc.clean), file = 'genes.bed', 
              row.names = FALSE, col.names = FALSE, quote = FALSE)

write.table(do.call(rbind,exons.bed.proc.clean), file = 'exons.bed', 
              row.names = FALSE, col.names = FALSE, quote = FALSE)

write.table(do.call(rbind,genes.bed.padded), file = 'genes_padded.bed', 
              row.names = FALSE, col.names = FALSE, quote = FALSE)

# Redo map.to.xmap
files.split = tstrsplit(dir(),'.',fixed = TRUE)
map.files = dir()[files.split[[2]] == 'map']
bed.files = dir()[files.split[[2]] == 'bed'] # the previous batch of '.bed' files are in subfolders, with different extensions
bed.names = tstrsplit(bed.files,'.',fixed = TRUE)[[1]]
ptm = proc.time()
for (i in map.files) { 
  xmap.filename = strsplit(i,'.',fixed = TRUE)[[1]][1] # accidentally overwrote XMAPs from the first run on my machine
  xmap.filename = paste0(xmap.filename, '-genes.xmap')
  map.to.xmap(i, bed.files, bed.names, xmap.file = xmap.filename); print(proc.time() - ptm) } 

# Write SNPs - copied and modified from script used to find accessible SNPs
xmap.names = dir()[grep('-genes.xmap',dir(), fixed = TRUE)]
xmaps = lapply(xmap.names, fread)
xmaps.merged = do.call(rbind, xmaps)

write.table(xmaps.merged[exons == 1,]$snp, file = 'SNPs_in_exons.txt', sep = '\t', col.names = FALSE, row.names = FALSE, quote = FALSE)
write.table(xmaps.merged[genes == 1,]$snp, file = 'SNPS_in_genes.txt', sep = '\t', col.names = FALSE, row.names = FALSE, quote = FALSE)
write.table(xmaps.merged[genes_padded == 1,]$snp, file = 'SNPS_in_padded_genes.txt', sep = '\t', col.names = FALSE, row.names = FALSE, quote = FALSE)

