#!/usr/bin/env/Rscript

#  Analyze fixations in quadrants (location bias) in visual displays.
#
#  Usage: quadrants.r -f quadrants.csv [-d DELIMITER]
#
#  The input CSV file must have these columns:
#  image, username, quadrant
#
#  External dependencies, to be installed with `install.packages(..., deps=T)` within R.
#  - none
#
#  Authors:
#  - Luis A. Leiva <name.surname@uni.lu>

library(ggplot2)
library(optparse)


cli_options = list(
  make_option(c("-f", "--file"), type="character", default=NULL, help="dataset CSV file", metavar="filename"),
  make_option(c("-d", "--delim"), type="character", default=",", help="CSV file delimiter [default=%default]", metavar="string"),
  make_option(c("-p", "--plots"), action="store_true", default=FALSE, help="generate plots"),
  make_option(c("-t", "--title"), type="character", default=NULL, help="plot title", metavar="string"),
  make_option(c("-W", "--width"), type="numeric", default=8, help="plot width, in inches [default=%default]", metavar="number"),
  make_option(c("-H", "--height"), type="numeric", default=8, help="plot height, in inches [default=%default]", metavar="number")
)
opt_parser = OptionParser(option_list=cli_options)
opt = parse_args(opt_parser)


prop_report = function(chiSq, N) {
  phi = sqrt(chiSq$statistic/N)
  cat(sprintf('\n\\chi^2_{(%s, N=%.2f)} = %.3f, p = %.4f, \\phi = %.3f\n', chiSq$parameter, N, chiSq$statistic, chiSq$p.value, phi))
}

dat = read.csv(opt$file, sep=opt$delim, header=T)

# Aggregated data: num fixations (length) grouped by user and quadrant.
# Each fixation cannot be considered an independent observation, since they're repeated measures.
agg = aggregate(image ~ quadrant + username, dat, length)

q1s = subset(agg, agg$quadrant == 'q1')
q2s = subset(agg, agg$quadrant == 'q2')
q3s = subset(agg, agg$quadrant == 'q3')
q4s = subset(agg, agg$quadrant == 'q4')

# After aggregation, the `image` column has the fixation counts.
# So take the mean (since we have aggregated measures) and test our hypotheses; e.g.
# Were there more fixations on q1 vs q2 on average?
num_fixations = c(mean(q1s$image), mean(q2s$image), mean(q3s$image), mean(q4s$image))
cat('Q1 & Q2 & Q3 & Q4\n')
cat(sprintf('%.2f $\\pm$ %.2f & %.2f $\\pm$ %.2f & %.2f $\\pm$ %.2f & %.2f $\\pm$ %.2f \n',
  mean(q1s$image), sd(q1s$image), mean(q2s$image), sd(q2s$image), mean(q3s$image), sd(q3s$image), mean(q4s$image), sd(q4s$image)))

N = sum(num_fixations)
num_observations = rep.int(N, length(num_fixations))

prop_report(prop.test(num_fixations, num_observations), N)
#  \chi^2_{(3, N=528.16)} = 148.844, p = 0.0000, \phi = 0.531

pairwise.prop.test(num_fixations, num_observations)
#	  Pairwise comparisons using Pairwise comparison of proportions
#  data:  num_fixations out of num_observations
#    1       2       3
#  2 1.2e-12 -       -
#  3 0.18461 < 2e-16 -
#  4 0.00075 < 2e-16 0.04588


# Exit early if no plots are to be generated.
if (!opt$plots) q();

# Create color boxplots.
p = ggplot(agg, aes(x=quadrant, y=image)) +
  geom_boxplot(outlier.size=-1) +
  scale_x_discrete(labels=c("Q1", "Q2", "Q3", "Q4")) +
  theme(axis.title.x=element_blank(), text=element_text(size=30), axis.title.y=element_text(margin=margin(t=0, r=30, b=0, l=0))) +
  ggtitle(opt$title) + labs(y="Num fixations")

outfile = paste0(opt$file, '_boxplot.pdf')
ggsave(outfile, plot=last_plot(), width=opt$width, height=opt$height)
