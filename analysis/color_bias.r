#!/usr/bin/env/Rscript

#  Analyze color bias in visual displays.
#
#  Usage: color_bias.r -a all-colors.csv -f fixated-colors.csv [-d DELIMITER]
#
#  The input CSV file must have these columns:
#  frequency, r, g, b
#
#  External dependencies, to be installed with `install.packages(..., deps=T)` within R.
#  - ggplot2
#  - latex2exp
#
#  Authors:
#  - Luis A. Leiva <name.surname@uni.lu>

library(optparse)
library(ggplot2)
library(latex2exp)


cli_options = list(
  make_option(c("-i", "--image"), type="character", default=NULL, help="image colors CSV file", metavar="filename"),
  make_option(c("-f", "--fixated"), type="character", default=NULL, help="fixated colors CSV file", metavar="filename"),
  make_option(c("-d", "--delim"), type="character", default=",", help="CSV file delimiter [default=%default]", metavar="string"),
  make_option(c("-p", "--plots"), action="store_true", default=FALSE, help="generate plots"),
  make_option(c("-W", "--width"), type="numeric", default=8, help="plot width, in inches [default=%default]", metavar="number"),
  make_option(c("-H", "--height"), type="numeric", default=8, help="plot height, in inches [default=%default]", metavar="number")
)
opt_parser = OptionParser(option_list=cli_options)
opt = parse_args(opt_parser)


luma = function(df) {
  L = df$r * 0.2126 + df$g * 0.7152 + df$b * 0.0722
  L
}


colors_all = read.csv(opt$image, sep=opt$delim, header=T)
colors_fix = read.csv(opt$fixated, sep=opt$delim, header=T)

colors_all$brightness = luma(colors_all)
colors_fix$brightness = luma(colors_fix)
colors_all$group = 'non-fixation colors'
colors_fix$group = 'colors fixated on'

# Run statistical tests.
combined = rbind(colors_all, colors_fix)
bartlett.test(combined$brightness, combined$group)
cat('---\n')

weighted_color = combined$brightness * combined$frequency
bartlett.test(weighted_color, combined$group)
cat('---\n')

# Compute Cohen's d effect size.
c_all = colors_all$brightness
c_fix = colors_fix$brightness
d_unweighted = abs(mean(c_all) - mean(c_fix)) / sqrt((sd(c_all) + sd(c_fix))/2)

c_all = colors_all$brightness * colors_all$frequency
c_fix = colors_fix$brightness * colors_fix$frequency
d_weighted = abs(mean(c_all) - mean(c_fix)) / sqrt((sd(c_all) + sd(c_fix))/2)

cat(sprintf('Effect size: unweighted: %.3f, weighted: %.3f\n', d_unweighted, d_weighted))
cat('---\n')

# Exit early if no plots are to be generated.
if (!opt$plots) q();

# Create color boxplots.
p = ggplot(combined, aes(x=group, y=brightness)) +
  geom_boxplot(outlier.size=-1) +
  scale_y_continuous(limits=quantile(combined$brightness, c(0.1, 0.9))) +
  theme(axis.title.x=element_blank(), text=element_text(size=30), axis.title.y=element_text(margin=margin(t=0, r=30, b=0, l=0))) +
  labs(y=TeX("Color brightness ($cd/m^2$)"))

outfile = paste0(opt$image, '_boxplot_unweighted.pdf')
ggsave(outfile, plot=last_plot(), width=opt$width, height=opt$height)

p = ggplot(combined, aes(x=group, y=brightness*frequency)) +
  geom_boxplot(outlier.size=-1) +
  scale_y_continuous(limits=quantile(combined$brightness * combined$frequency, c(0.1, 0.9))) +
  theme(axis.title.x=element_blank(), text=element_text(size=30), axis.title.y=element_text(margin=margin(t=0, r=30, b=0, l=0))) +
  labs(y=TeX("Weighted color brightness ($cd/m^2$)"))

outfile = paste0(opt$fixated, '_boxplot_weighted.pdf')
ggsave(outfile, plot=last_plot(), width=opt$width, height=opt$height)
