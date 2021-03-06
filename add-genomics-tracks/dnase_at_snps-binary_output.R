# Papers in this field don't emphasize the amount of DNAse accessibility there is at a site - 
# they focus on whether a site is accessible or not (in other words, they treat accesibility as binary).
# This suggests that getting deeply into precise signal values for each hotspot is a form of over-interpretation.
# For now, I'll follow this precedent. For each cell type-SNP pair, output 1 if accessible (in an ENCODE hotspot),
# 0 if not.
# Papers: http://genome.cshlp.org/content/22/9/1711.long
# http://www.nature.com/nature/journal/v489/n7414/extref/nature11232-s1.pdf
# See also how Basset built their training data: http://genome.cshlp.org/content/26/7/990.full.pdf .
# Don't think we want to follow the Basset peak-merging approach though; let's just look up the SNPs.

# Possible Data: http://ftp.ebi.ac.uk/pub/databases/ensembl/encode/integration_data_jan2011/byDataType/openchrom/jan2011/fdrPeaks/
# or https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE32970 ?

# 2016-11-07: used https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE32970 ,
# picked 'Custom Archive,' downloaded all NARROWPEAK files, extracted.

# Ignoring position of peak for now - 
# for each feature in in.map, check if it's inside a feature in in.dat .
# if so, assign it a value equal to the corresponding signalValue in in.dat;
# else assign it 0.

require(data.table)
require(IRanges)

map.to.xmap = function(map.file, bed.files, bed.names, xmap.file, chr.map = function(x) paste0('chr',x)) {
 if(missing(bed.names)) { bed.names = bed.files } # 'bed' is a general name for the class of files NarrowPeak is in.
 if(missing(xmap.file)) { xmap.file = paste0(strsplit(map.file,'.', fixed = TRUE)[[1]][1], '.xmap') }
  in.map = fread(map.file); setnames(in.map, c('chr','snp','dist','pos'))
  chr.to.use = chr.map(in.map$chr[1]) # assume whole map file from same chromosome
  print(chr.to.use)
  for(i in 1:length(bed.files)) {
    print(bed.names[i])
    in.dat = fread(bed.files[i])
    names.new = c('chrom','chromStart','chromEnd','name','score','strand','signalValue','pValue','qValue','peak')[1:ncol(in.dat)]
    setnames(in.dat,names(in.dat),names.new)
    in.dat = in.dat[chrom == chr.to.use,]
    intervals = as.vector(rbind(in.dat$chromStart, in.dat$chromEnd))
    in.interv = function(x, intervals) {
      tmp = findInterval(x, intervals)
      tmp = tmp %% 2
      return(tmp)
    }
    in.map[,bed.names[i] := in.interv(pos, intervals) ]
  }
  # Note: output file keeps header names; we don't want to lose info about which cell type is which.
  # This is different from the original .map file format, which has no header.
  write.table(in.map, file = xmap.file, sep = '\t', row.names = FALSE, quote = FALSE)
}

# For 2016-12-04 work with gene .bed files, which UCSC browser provides with redundant entries
# and possibly overlapping entries (and adding padding could introduce more overlaps)
# REQUIRES THAT ONLY ONE CHROMOSOME IS PRESENT

clean.bed = function(bed.in) {
  if(length(unique(bed.in$chrom)) != 1) { 
      print(paste0('Wrong number of chromosomes: ',unique(bed.in$chrom), collapse = ''))
      return()
  }
  # Remove duplicate start-end pairs
  interval.pairs = paste0(bed.in$chromStart,'-',bed.in$chromEnd)
  interval.pairs.u = unique(interval.pairs)
  interval.pairs.s = sapply(interval.pairs.u, function(x,y) which(y %in% x)[1], interval.pairs)
  bed.in = bed.in[interval.pairs.s,]
  bed.in.ir = IRanges::IRanges(start = bed.in$chromStart, end = bed.in$chromEnd)
  bed.in.ir = IRanges::reduce(bed.in.ir)
  ans = data.table(chrom = rep(bed.in$chrom[1],length(bed.in.ir)), 
                   chromStart = IRanges::start(bed.in.ir),
                   chromEnd = IRanges::end(bed.in.ir))
  return(ans)
}

# intervals.in is a data.table with columns 'chrom', 'chromStart', 'chromEnd'
# pad.size is a positive integer
# Add padding of pad.size to each gene, at both start and end.
# Some may technically reach beyond the start/end of the chromosome;
# will leave these be, shouldn't hurt for what we're doing.
# Use 'clean.bed' to resolve any overlaps introduced.
pad.genes = function(intervals.in, pad.size) {
  intervals.use = data.table::copy(intervals.in) # avoid goofy pass-by-reference behavior
  intervals.use[,chromStart := chromStart - pad.size]
  intervals.use[,chromEnd := chromEnd + pad.size]
  return(clean.bed(intervals.use))
}

