# Load all .xmap files and rbind them; strip redundant headers
# set RS numbers as row names; strip all header columns
# apply across to get counts
# Output files: one with all RS numbers and with counts, one each for  >1 and >0 counts
require(data.table)
setwd('C:/Users/Ben/Documents/CS 273B project')
xmap.names = dir()[grep('.xmap',dir(), fixed = TRUE)]
xmaps = lapply(xmap.names, fread)
xmaps.merged = do.call(rbind, xmaps)
snps = xmaps.merged$snp
xmaps.merged[,c('chr','snp','dist','pos') := NULL]
snp.cts = apply(xmaps.merged,1,sum)
snp.cts.all = cbind(snps, snp.cts)
snp.cts.2 = cbind(snps, snp.cts)[snp.cts >= 2,]
snp.cts.1 = cbind(snps, snp.cts)[snp.cts >= 1,]

write.table(snp.cts.all, file = 'SNP_occurrence_counts.txt', sep = '\t', col.names = TRUE, row.names = FALSE, quote = FALSE)
write.table(snp.cts.2[,1], file = 'SNPs_morethan1.txt', sep = '\t', col.names = FALSE, row.names = FALSE, quote = FALSE)
write.table(snp.cts.1[,1], file = 'SNPS_morethan0.txt', sep = '\t', col.names = FALSE, row.names = FALSE, quote = FALSE)