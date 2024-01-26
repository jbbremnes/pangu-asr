#  make plots of verification statistics

using JLD2, DataFrames, Statistics
using Plots
using LaTeXStrings


#  get verification statistics
JLD2.@load "./data/scores_bqn_+60.jld2"  # => scores
JLD2.@load "./data/scores_raw_+60.jld2"  # => scores_raw

@assert scores.models == scores_raw.models
@assert scores.lts == scores_raw.lts
models = scores.models
lts    = scores.lts


#  define colour palette
mypalette = Dict("pangu" => :black, "hres" => :red, "meps0" => :lightblue, "ens0" => :lightgreen,
                 "meps" => :dodgerblue3, "ens" => :darkseagreen)
mypalette = (pangu = :black, hres = :red, meps0 = :lightblue, ens0 = :lightgreen,
             meps = :dodgerblue3, ens = :darkseagreen)
#cols      = [mypalette[m] for m in models]
cols = collect(mypalette)
labels = ["PANGU" "HRES" "MEPS0" "ENS0" "MEPS" "ENS"]
attr = (fontsize = 14, lwd = 2)



#
#    R A W 
#

#  temperature scores
p1 = plot(scores_raw.lts, scores_raw.mae[:,:,1],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "mean absolute error (K)",
          linewidth = attr.lwd, palette = cols,
          legend = :topleft, labels = labels, legendcolumns = 2, fg_legend = :transparent,
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize-1, guidefontsize= attr.fontsize,
          titleposition=:center)
p2 = plot(scores_raw.lts, scores_raw.bias[:,:,1],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "mean error (K)",
          linewidth = attr.lwd, palette = cols,
          legend = false, 
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize, guidefontsize= attr.fontsize,
          titleposition=:center)
p3 = plot(scores_raw.lts, scores_raw.sde[:,:,1],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "standard deviation of error (K)",
          linewidth = attr.lwd, palette = cols, legend = false,
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize, guidefontsize= attr.fontsize,
          titleposition=:center)
p = plot(p1, p2, p3, layout= (1,3), size = (1800,600), dpi = 300,
         left_margin = (10,:px), bottom_margin = (40,:px))
savefig(p, "./data/plots/t2m_raw_scores.png")

p = plot(p1, p2, p3, layout= (1,3), size = (1800,600), dpi = 300,
         left_margin = (5,:mm), bottom_margin = (10,:mm))
savefig(p, "./data/plots/t2m_raw_scores.pdf")


#  temperature forecast properties
p1 = plot(scores_raw.lts, scores_raw.sd[:,:,1],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "standard deviation ratio",
          linewidth = attr.lwd, palette = cols,
          legend = :topleft, labels = labels, legendcolumns = 2, fg_legend = :transparent,
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize-1, guidefontsize= attr.fontsize,
          titleposition=:center)
p2 = plot(scores_raw.lts, scores_raw.dmax[:,:,1],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "mean difference in maxima (K)",
          linewidth = attr.lwd, palette = cols,
          legend = false, 
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize, guidefontsize= attr.fontsize,
          titleposition=:center)
p3 = plot(scores_raw.lts, scores_raw.dmin[:,:,1],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "mean difference in minima (K)",
          linewidth = attr.lwd, palette = cols, legend = false,
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize, guidefontsize= attr.fontsize,
          titleposition=:center)
p = plot(p1, p2, p3, layout= (1,3), size = (1800,600), dpi = 300,
         left_margin = (5,:mm), bottom_margin = (10,:mm))
savefig(p, "./data/plots/t2m_raw_var.pdf")


# temperature single figure
p = [plot(scores_raw.lts, scores_raw.mae[:,:,1],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "mean absolute error (K)",
          linewidth = attr.lwd, palette = cols,
          legend = :topleft, labels = labels, legendcolumns = 2, fg_legend = :transparent,
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize-1, guidefontsize= attr.fontsize,
          titleposition=:center),
     plot(scores_raw.lts, scores_raw.bias[:,:,1],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "mean error (K)",
          linewidth = attr.lwd, palette = cols,
          legend = false, 
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize, guidefontsize= attr.fontsize,
          titleposition=:center),
     plot(scores_raw.lts, scores_raw.sde[:,:,1],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "standard deviation of error (K)",
          linewidth = attr.lwd, palette = cols, legend = false,
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize, guidefontsize= attr.fontsize,
          titleposition=:center),
     plot(scores_raw.lts, scores_raw.sd[:,:,1],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "standard deviation ratio",
          linewidth = attr.lwd, palette = cols,
          legend = false,
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize-1, guidefontsize= attr.fontsize,
          titleposition=:center),
     plot(scores_raw.lts, scores_raw.dmax[:,:,1],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "deviation in maxima (K)",
          linewidth = attr.lwd, palette = cols,
          legend = false, 
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize, guidefontsize= attr.fontsize,
          titleposition=:center),
     plot(scores_raw.lts, scores_raw.dmin[:,:,1],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "deviation in minima (K)",
          linewidth = attr.lwd, palette = cols, legend = false,
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize, guidefontsize= attr.fontsize,
          titleposition=:center)
     ]

pout = plot(p..., layout= (2,3), size = (1800,1000), dpi = 300,
            left_margin = (5,:mm), top_margin = (2,:mm), bottom_margin = (12,:mm))
savefig(pout, "./data/plots/t2m_raw.pdf")


# wind speed single figure
prm = 2
p = [plot(scores_raw.lts, scores_raw.mae[:,:,prm],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "mean absolute error (ms⁻¹)",
          linewidth = attr.lwd, palette = cols,
          legend = :left, labels = labels, legendcolumns = 2, fg_legend = :transparent,
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize-1, guidefontsize= attr.fontsize,
          titleposition=:center),
     plot(scores_raw.lts, scores_raw.bias[:,:,prm],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "mean error (ms⁻¹)",
          linewidth = attr.lwd, palette = cols,
          legend = false, 
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize, guidefontsize= attr.fontsize,
          titleposition=:center),
     plot(scores_raw.lts, scores_raw.sde[:,:,prm],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "standard deviation of error (ms⁻¹)",
          linewidth = attr.lwd, palette = cols, legend = false,
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize, guidefontsize= attr.fontsize,
          titleposition=:center),
     plot(scores_raw.lts, scores_raw.sd[:,:,prm],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "standard deviation ratio",
          linewidth = attr.lwd, palette = cols,
          legend = false,
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize-1, guidefontsize= attr.fontsize,
          titleposition=:center),
     plot(scores_raw.lts, scores_raw.dmax[:,:,prm],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "deviation in maxima (ms⁻¹)",
          linewidth = attr.lwd, palette = cols,
          legend = false, 
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize, guidefontsize= attr.fontsize,
          titleposition=:center),
     plot(scores_raw.lts, scores_raw.rmax[:,:,prm],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "maxima ratio",
          linewidth = attr.lwd, palette = cols, legend = false,
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize, guidefontsize= attr.fontsize,
          titleposition=:center)
     ]

pout = plot(p..., layout= (2,3), size = (1800,1000), dpi = 300,
            left_margin = (5,:mm), top_margin = (2,:mm), bottom_margin = (12,:mm))
savefig(pout, "./data/plots/ws10m_raw.pdf")






#
#     P R O B A B I L I S T I C
#


sc_crps = dropdims(mean(scores.crps, dims=2:3), dims=(2,3))
sc_qs   = dropdims(mean(scores.qs, dims=2:3), dims=(2,3))
sc_mae  = dropdims(mean(scores.mae, dims=2:3), dims=(2,3))
    
#  temperature
prm = 1
p1 = plot(scores.lts, sc_crps[:,:,prm],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "continuous ranked probability score (K)",
          linewidth = attr.lwd, palette = cols,
          legend = :bottomright, labels = labels, legendcolumns = 2, fg_legend = :transparent,
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize-1, guidefontsize= attr.fontsize,
          titleposition=:center)
p2 = plot(scores.lts, sc_mae[:,:,prm],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "mean absolute error (K)",
          linewidth = attr.lwd, palette = cols,
          legend = :false,
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize-1, guidefontsize= attr.fontsize,
          titleposition=:center)
#crps_all = hcat(sc_crps[:,:,prm], scores_raw.crps[:,5:6,prm])
p3 = plot(scores.lts, sc_crps[:,:,prm],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "continuous ranked probability score (K)",
          linewidth = attr.lwd, palette = cols,
          legend = false,
          #legend = :left, labels = [labels;labels], legendcolumns = 4, fg_legend = :transparent,
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize-1, guidefontsize= attr.fontsize,
          titleposition=:center)
plot!(scores_raw.lts, scores_raw.crps[:,5:6,prm],
      linewidth = attr.lwd, palette = cols[5:6], linestyle = :dash)
annotate!([(6, 1.4, text("dashed lines:\nraw forecast models", 12, :left))])
p4 = plot(scores.lts, sc_mae[:,:,prm],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "mean absolute error (K)",
          linewidth = attr.lwd, palette = cols,
          #legend = :topleft, labels = labels, legendcolumns = 2, fg_legend = :transparent,
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize-1, guidefontsize= attr.fontsize,
          titleposition=:center)
plot!(scores_raw.lts, scores_raw.mae[:,:,prm],
      linewidth = attr.lwd, palette = cols, linestyle = :dash, legend = false)

pout = plot(p1, p2, p3, p4, layout = (2,2), size = (1200,1000), dpi = 300,
            left_margin = (5,:mm), top_margin = (2,:mm), bottom_margin = (5,:mm))
savefig(pout, "./data/plots/t2m_prob.pdf")

#  wind speed
prm = 2
p1 = plot(scores.lts, sc_crps[:,:,prm],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "continuous ranked probability score (ms⁻¹)",
          linewidth = attr.lwd, palette = cols,
          legend = :topleft, labels = labels, legendcolumns = 2, fg_legend = :transparent,
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize-1, guidefontsize= attr.fontsize,
          titleposition=:center)
p2 = plot(scores.lts, sc_mae[:,:,prm],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "mean absolute error (ms⁻¹)",
          linewidth = attr.lwd, palette = cols,
          legend = :false,
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize-1, guidefontsize= attr.fontsize,
          titleposition=:center)
#crps_all = hcat(sc_crps[:,:,prm], scores_raw.crps[:,5:6,prm])
p3 = plot(scores.lts, sc_crps[:,:,prm],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "continuous ranked probability score (ms⁻¹)",
          linewidth = attr.lwd, palette = cols,
          legend = false,
          #legend = :left, labels = [labels;labels], legendcolumns = 4, fg_legend = :transparent,
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize-1, guidefontsize= attr.fontsize,
          titleposition=:center)
plot!(scores_raw.lts, scores_raw.crps[:,5:6,prm],
      linewidth = attr.lwd, palette = cols[5:6], linestyle = :dash)
annotate!([(6, 1.5, text("dashed lines:\nraw forecast models", 12, :left))])
p4 = plot(scores.lts, sc_mae[:,:,prm],
          xticks = scores_raw.lts,
          xlabel = "lead time (h)", ylabel = "", title = "mean absolute error (ms⁻¹)",
          linewidth = attr.lwd, palette = cols,
          #legend = :topleft, labels = labels, legendcolumns = 2, fg_legend = :transparent,
          titlefontsize = attr.fontsize, tickfontsize = attr.fontsize,
          legendfontsize = attr.fontsize-1, guidefontsize= attr.fontsize,
          titleposition=:center)
plot!(scores_raw.lts, scores_raw.mae[:,:,prm],
      linewidth = attr.lwd, palette = cols, linestyle = :dash, legend = false)

pout = plot(p1, p2, p3, p4, layout = (2,2), size = (1200,1000), dpi = 300,
            left_margin = (5,:mm), top_margin = (2,:mm), bottom_margin = (5,:mm))
savefig(pout, "./data/plots/ws10m_prob.pdf")


















