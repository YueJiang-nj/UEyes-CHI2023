#!/usr/bin/env/Rscript

#  Analyze location bias in visual displays.
#
#  Usage: location_bias.r -f fixations.csv [-d DELIMITER]
#
#  The input CSV file must have these columns:
#  image, width, height, username, x, y, timestamp, duration
#
#  External dependencies, to be installed with `install.packages(..., deps=T)` within R.
#  - ggplot2
#  - latex2exp
#
#  Authors:
#  - Luis A. Leiva <name.surname@uni.lu>

library(optparse)
library(ggplot2)
library(gridExtra)


cli_options = list(
  make_option(c("-f", "--file"), type="character", default=NULL, help="dataset CSV file", metavar="filename"),
  make_option(c("-d", "--delim"), type="character", default=",", help="CSV file delimiter [default=%default]", metavar="string"),
  make_option(c("-W", "--width"), type="numeric", default=8, help="plot width, in inches [default=%default]", metavar="number"),
  make_option(c("-H", "--height"), type="numeric", default=5, help="plot height, in inches [default=%default]", metavar="number")
)
opt_parser = OptionParser(option_list=cli_options)
opt = parse_args(opt_parser)

# Rescale values in the [0,1] range.
rescale <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# Remap values in the [from,to] range.
linMap <- function(x, from, to) {
  rescale(x) * (to - from) + from
}


# Load dataset.
dat = read.csv(opt$file, sep=opt$delim, header=T)

# TODO: Ignore fixations outside screenshot viewport?
# In the experiment, all images were fit to the max height of the viewing monitor,
# therefore in many cases there were horizontal strips (black space).
# However, for a location analysis this might not be really needed,
# so let's assume that each image fitted the available screen space.
ui_width = 1920
ui_height = 1200

# Instead of linear mapping, make fixations relative to the new viewport (elastic mapping).
# since e.g. `linMap(dat$x, 0, ui_width)` will *offset* the X coords to [0,ui_width].
#dat$x = dat$x * ui_width / dat$width
#dat$y = dat$y * ui_height / dat$height

# Better: Center coords in screen.
dat$x = dat$x + (ui_width - dat$width) / 2
dat$y = dat$y + (ui_height - dat$height) / 2


# Helper plot function.
my_scale_y_continuous = function(val) {
  scale_y_continuous(breaks=seq(0, 0.9*max(density(val)$y), len=2), labels=function(v)sprintf("%.3f", v))
}

# Compose histogram: top, center, right sub-plots.
hist_top = ggplot(dat, aes(x=x)) +
             stat_density(geom="line", size=1, color='darkred') +
             labs(x='fixations along X axis') +
             geom_vline(aes(xintercept=ui_width/2), color="gray", linetype="dashed", size=1) +
             scale_x_continuous(limits=c(0, ui_width), expand=c(0,0)) +
             my_scale_y_continuous(dat$x)

hist_center = ggplot(dat, aes(x=x, y=y)) +
                geom_density_2d_filled() +
                labs(x='screen width', y='screen height') +
                geom_vline(aes(xintercept=ui_width/2), color="gray", linetype="dashed", size=1) +
                geom_hline(aes(yintercept=ui_height/2), color="gray", linetype="dashed", size=1) +
                scale_x_continuous(limits=c(0, ui_width), expand=c(0,0)) +
                scale_y_reverse(limits=c(ui_height, 0), expand=c(0,0)) +
                theme(legend.position="none", panel.background=element_blank(),
                  axis.text.y=element_text(margin=margin(t=0, r=0, b=0, l=10)),
                  panel.border=element_rect(color="black", fill=NA, size=2),
                  axis.line=element_line(color="black"))

hist_right = ggplot(dat, aes(x=y)) +
               stat_density(geom="line", size=1, color='darkblue') +
               labs(x='fixations along Y axis') +
               geom_vline(aes(xintercept=ui_height/2), color="gray", linetype="dashed", size=1) +
               coord_flip() +
               scale_x_reverse(limits=c(ui_height, 0), expand=c(0,0)) +
               my_scale_y_continuous(dat$y)

empty = ggplot() + geom_point(aes(1,1), color="white") +
        theme(axis.ticks=element_blank(), panel.background=element_blank(),
              axis.text.x=element_blank(), axis.text.y=element_blank(),
              axis.title.x=element_blank(), axis.title.y=element_blank())

outfile = paste0(opt$file, '_location_bias.pdf')
pdf(outfile, width=opt$width, height=opt$height)
#  Layout grid: 2 cols by 2 rows.
#  +-------+---+
#  |       | e | <- empty cell
#  |-------|---|
#  |       |   |
#  |       |   |
#  +-------+---+
grid.arrange(
  hist_top, empty,
  hist_center, hist_right,
  ncol=2,
  widths=c(3, 1),
  nrow=2,
  heights=c(1, 2)
)
dev.off()
