using CairoMakie
using Statistics: mean

# =============================================================================
# CONFIG
# =============================================================================
# Mirrors the hyperparameters in train_potts.jl / train_binary.jl so filenames
# can be reconstructed exactly. Same ARGS convention as the training scripts:
# N_ITERS PAIRED_ITERS [H_ADD=30] [N_HIDDEN=30] [REG=0] [DATASET_TAG=20links_09strength] [HID_TAG=] [SEED=42]
const OUTPUT_DIR   = "./results"
const CD_STEPS     = 50
const LOG_EVERY    = 100
const N_ITERS      = parse(Int, ARGS[1])
const PAIRED_ITERS = parse(Int, ARGS[2])
const H_ADD        = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 30
const N_HIDDEN     = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 30
const REG          = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 0
const DATASET_TAG  = length(ARGS) >= 6 ? ARGS[6] : "20links_09strength"

regstr(r) = isinteger(r) ? string(Int(r)) : string(r)  # REG=0 (not 0.0) matches pre-existing filenames
# Hidden-unit type for the Potts side (must match whatever tag, if any,
# train_potts.jl used when it wrote these files). Defaults to empty so this
# script keeps reading the many pre-existing nsReLU-era runs, which predate
# this tag entirely and have no "_HID=" in their filenames at all — only pass
# a 7th arg (e.g. "xReLU") when analyzing a run trained with that tag.
const HID_TAG = length(ARGS) >= 7 ? ARGS[7] : ""
hidsuffix() = isempty(HID_TAG) ? "" : "_HID=$(HID_TAG)"
potts_label = isempty(HID_TAG) ? "Potts+nsReLU" : "Potts+$(HID_TAG)"

# Same default-42/only-tag-if-different convention as train_potts.jl's SEED.
const SEED = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 42
seedsuffix() = SEED == 42 ? "" : "_SEED=$(SEED)"

# Same convention for the learning rate.
const LR = length(ARGS) >= 9 ? parse(Float64, ARGS[9]) : 1e-3
lrsuffix() = LR == 1e-3 ? "" : "_LR=$(LR)"

const PARAMTAG = "N_HIDDEN=$(N_HIDDEN)_H_ADD=$(H_ADD)_N_ITERS=$(N_ITERS)_PAIRED_ITERS=$(PAIRED_ITERS)_REG=$(regstr(REG))_DATA=$(DATASET_TAG)$(hidsuffix())$(seedsuffix())$(lrsuffix())"
const REPORT_DIR = joinpath(OUTPUT_DIR, "analysis_$(PARAMTAG)")
isdir(REPORT_DIR) || mkpath(REPORT_DIR)

# Every figure filename carries the full param tag too (not just the
# containing directory) so files don't get confused once copied elsewhere or
# viewed in a flat gallery alongside other runs.
figpath(name) = joinpath(REPORT_DIR, "$(name)_$(PARAMTAG).png")

const ARCHS = [("" , potts_label), ("binary_", "BinaryRBM")]

# Private RBMs (A, B) are H_ADD-independent — train_potts.jl now saves/reuses
# them under a suffix without H_ADD, so lookups here must match.
private_suffix() = "N_HIDDEN=$(N_HIDDEN)_K=$(CD_STEPS)_N_ITERS=$(N_ITERS)_REG=$(regstr(REG))_DATA=$(DATASET_TAG)$(hidsuffix())$(seedsuffix())$(lrsuffix())"
single_suffix() = "N_HIDDEN=$(N_HIDDEN)_H_ADD=$(H_ADD)_K=$(CD_STEPS)_N_ITERS=$(N_ITERS)_REG=$(regstr(REG))_DATA=$(DATASET_TAG)$(hidsuffix())$(seedsuffix())$(lrsuffix())"
paired_suffix()  = single_suffix() * "_PAIRED_ITERS=$(PAIRED_ITERS)"
path(arch, name, suffix) = joinpath(OUTPUT_DIR, "$(arch)$(name)_$(suffix).txt")

# =============================================================================
# LINE PARSERS
# =============================================================================
# All checkpoint files were written by our own `println`-based loggers in
# train_potts.jl/train_binary.jl, so the format is fixed and known — this
# reconstructs the numbers directly from those exact "key=value" text lines
# rather than re-deriving anything from the raw hdf5 models.
extract_iter(line) = parse(Int, match(r"\biter=(\d+)", line).captures[1])

function extract_scalar(line, key)
    m = match(Regex("\\b" * key * "=([^\\s]+)"), line)
    isnothing(m) && return NaN
    # normalize Float32 "f-exponent" scientific notation (e.g. "6.76f-5") to
    # standard "e-exponent" notation, which Base.parse actually accepts.
    return parse(Float64, replace(m.captures[1], r"f(-?\d+)$" => s"e\1"))
end

function extract_array(line, key)
    m = match(Regex("\\b" * key * "=(?:\\w+)?\\[([^\\]]*)\\]"), line)
    (isnothing(m) || isempty(strip(m.captures[1]))) && return Float64[]
    # small (e.g. regularized-toward-zero) Float32 values print in "f-exponent"
    # scientific notation (e.g. "6.76f-5"), which Base.parse doesn't accept —
    # normalize to standard "e-exponent" notation first.
    vals = replace.(strip.(split(m.captures[1], ",")), r"f(-?\d+)$" => s"e\1")
    return parse.(Float64, vals)
end

function extract_namedtuple(line, key)
    m = match(Regex("\\b" * key * "=\\(([^)]*)\\)"), line)
    isnothing(m) && return Dict{String,Float64}()
    out = Dict{String,Float64}()
    for pair in split(m.captures[1], ",")
        kv = split(pair, "=")
        length(kv) == 2 || continue
        v = replace(strip(kv[2]), r"f(-?\d+)$" => s"e\1")  # Float32 literal -> parseable
        out[strip(kv[1])] = parse(Float64, v)
    end
    return out
end

# =============================================================================
# FILE-TYPE LOADERS
# =============================================================================
function load_lpl(p)
    isfile(p) || return (iters=Int[], values=Float64[])
    vals = parse.(Float64, readlines(p))
    return (iters=collect(LOG_EVERY .* (1:length(vals))), values=vals)
end

function load_correlation(p)  # vh/v/ab_check common fields
    isfile(p) || return (iters=Int[], r=Float64[], slope=Float64[], intercept=Float64[],
                          mean_abs_diff=Float64[], max_abs_diff=Float64[])
    lines = readlines(p)
    getf(k) = [extract_scalar(l, k) for l in lines]
    return (iters=extract_iter.(lines), r=getf("r"), slope=getf("slope"), intercept=getf("intercept"),
            mean_abs_diff=getf("mean_abs_diff"), max_abs_diff=getf("max_abs_diff"))
end

function load_vh(p)
    base = load_correlation(p)
    isfile(p) || return merge(base, (mean_h_data=Vector{Float64}[], mean_h_model=Vector{Float64}[]))
    lines = readlines(p)
    return merge(base, (mean_h_data=[extract_array(l, "mean_h_data") for l in lines],
                         mean_h_model=[extract_array(l, "mean_h_model") for l in lines]))
end

function load_norm_check(p)  # wn/gn_check
    isfile(p) || return (iters=Int[], mean_norm=Float64[], max_norm=Float64[], min_norm=Float64[], norms=Vector{Float64}[])
    lines = readlines(p)
    getf(k) = [extract_scalar(l, k) for l in lines]
    return (iters=extract_iter.(lines), mean_norm=getf("mean_norm"), max_norm=getf("max_norm"),
            min_norm=getf("min_norm"), norms=[extract_array(l, "norms") for l in lines])
end

function load_firing(p)
    isfile(p) || return (iters=Int[], data_rate=Vector{Float64}[], model_rate=Vector{Float64}[])
    lines = readlines(p)
    return (iters=extract_iter.(lines), data_rate=[extract_array(l, "data_rate") for l in lines],
            model_rate=[extract_array(l, "model_rate") for l in lines])
end

function load_freeze(p)
    isfile(p) || return (iters=Int[], status=String[], max_before=Float64[], max_after=Float64[])
    lines = readlines(p)
    status = [String(match(r"\bstatus=(\S+)", l).captures[1]) for l in lines]
    before = [extract_namedtuple(l, "drift_before_reset") for l in lines]
    after  = [extract_namedtuple(l, "drift_after_reset") for l in lines]
    return (iters=extract_iter.(lines), status=status,
            max_before=[isempty(d) ? NaN : maximum(values(d)) for d in before],
            max_after=[isempty(d) ? NaN : maximum(values(d)) for d in after])
end

function load_architecture(arch)
    ss, ps = private_suffix(), paired_suffix()
    return (
        lpl_A         = load_lpl(path(arch, "lpl_A", ss)),
        lpl_B         = load_lpl(path(arch, "lpl_B", ss)),
        lpl_paired    = load_lpl(path(arch, "lpl_paired", ps)),
        lplval_A      = load_lpl(path(arch, "lplval_A", ss)),
        lplval_B      = load_lpl(path(arch, "lplval_B", ss)),
        lplval_paired = load_lpl(path(arch, "lplval_paired", ps)),
        vh_A          = load_vh(path(arch, "vh_check_A", ss)),
        vh_B          = load_vh(path(arch, "vh_check_B", ss)),
        vh_paired     = load_vh(path(arch, "vh_check_paired", ps)),
        v_A           = load_correlation(path(arch, "v_check_A", ss)),
        v_B           = load_correlation(path(arch, "v_check_B", ss)),
        v_paired      = load_correlation(path(arch, "v_check_paired", ps)),
        ab_paired     = load_correlation(path(arch, "ab_check_paired", ps)),
        firing_paired = load_firing(path(arch, "firing_check_paired", ps)),
        wn_A          = load_norm_check(path(arch, "wn_check_A", ss)),
        wn_B          = load_norm_check(path(arch, "wn_check_B", ss)),
        wn_paired     = load_norm_check(path(arch, "wn_check_paired", ps)),
        gn_A          = load_norm_check(path(arch, "gn_check_A", ss)),
        gn_B          = load_norm_check(path(arch, "gn_check_B", ss)),
        gn_paired     = load_norm_check(path(arch, "gn_check_paired", ps)),
        freeze_paired = load_freeze(path(arch, "freeze_check_paired", ps)),
    )
end

data = Dict(label => load_architecture(arch) for (arch, label) in ARCHS)

has(x) = !isempty(x.iters)

# tail-average over the last `frac` fraction of checkpoints (min 1 point) —
# used to summarize "where a metric ended up" without being noisy from a
# single last data point.
function tailmean(v::AbstractVector{<:Real}; frac=0.2)
    isempty(v) && return NaN
    n = max(1, round(Int, frac * length(v)))
    return mean(v[end-n+1:end])
end

println("Loaded architectures: ", join(last.(ARCHS), ", "))
for (_, label) in ARCHS
    d = data[label]
    println("  [$label] lpl_A n=$(length(d.lpl_A.iters))  lpl_paired n=$(length(d.lpl_paired.iters))  ab_paired n=$(length(d.ab_paired.iters))")
end

# =============================================================================
# PLOTS
# =============================================================================
colors = Dict(potts_label => :dodgerblue, "BinaryRBM" => :orangered)

# Every individual panel is forced to a square box via `aspect=1` (a plain
# number sets the *visual* box aspect, not a data-unit aspect — confirmed this
# doesn't distort line plots whose x/y ranges are wildly different, e.g.
# iteration counts in the thousands vs. a correlation in [0,1]). Figure canvas
# size is then just CELL px per panel times the grid's actual (rows, cols), so
# a 1x3 row of squares gets a wide-but-short canvas and a 2x1 column gets a
# tall-but-narrow one — no forced-square canvas fighting the per-panel shape.
const CELL = 480

# lines! alone is invisible for single-checkpoint series (e.g. short smoke-test
# runs with only one LOG_EVERY point) — always pair it with scatter! markers.
# `label` goes on the line only (scatter! would otherwise duplicate the legend
# entry); `linestyle` only applies to the line, not the markers.
function lines_and_points!(ax, x, y; color, linestyle=:solid, label=nothing)
    lines!(ax, x, y; color, linestyle, label)
    scatter!(ax, x, y; color, markersize=6)
end

function fig_lpl()
    fig = Figure(size=(3 * CELL, CELL))
    for (col, (model, tlabel)) in enumerate([(:A, "RBM A"), (:B, "RBM B"), (:paired, "Paired")])
        ax = Axis(fig[1, col]; title=tlabel, xlabel="iteration", ylabel=(col == 1 ? "log-pseudolikelihood" : ""), aspect=1)
        nplotted = 0
        for (_, label) in ARCHS
            d = data[label]
            train = getfield(d, Symbol("lpl_$model"))
            val   = getfield(d, Symbol("lplval_$model"))
            has(train) && (lines_and_points!(ax, train.iters, train.values; color=colors[label], linestyle=:solid, label="$label train"); nplotted += 1)
            has(val)   && (lines_and_points!(ax, val.iters, val.values; color=colors[label], linestyle=:dash, label="$label val"); nplotted += 1)
        end
        col == 3 && nplotted > 0 && axislegend(ax; position=:rb, framevisible=false, labelsize=10)
    end
    save(figpath("fig1_pseudolikelihood"), fig)
    return fig
end

function fig_alignment(checkname, getter, title_prefix)
    fig = Figure(size=(3 * CELL, 2 * CELL))
    for (col, (model, tlabel)) in enumerate([(:A, "RBM A"), (:B, "RBM B"), (:paired, "Paired")])
        axr = Axis(fig[1, col]; title=tlabel, ylabel=(col == 1 ? "r (correlation)" : ""), aspect=1)
        axs = Axis(fig[2, col]; xlabel="iteration", ylabel=(col == 1 ? "slope" : ""), aspect=1)
        hlines!(axr, [1.0]; color=:gray, linestyle=:dot)
        hlines!(axs, [1.0]; color=:gray, linestyle=:dot)
        nplotted = 0
        for (_, label) in ARCHS
            d = getter(data[label], model)
            if has(d)
                lines_and_points!(axr, d.iters, d.r; color=colors[label], label=label)
                lines_and_points!(axs, d.iters, d.slope; color=colors[label], label=label)
                nplotted += 1
            end
        end
        col == 3 && nplotted > 0 && axislegend(axr; position=:rb, framevisible=false, labelsize=10)
    end
    Label(fig[0, :], "$title_prefix — r/slope → 1 means model tracks data along y=x"; fontsize=16)
    save(figpath("fig_$(checkname)"), fig)
    return fig
end

function fig_ab_headline()
    fig = Figure(size=(CELL, 2 * CELL))
    axr = Axis(fig[1, 1]; title="A-B cross-family correlation alignment (paired RBM)",
               ylabel="r", xlabel="iteration", aspect=1)
    axs = Axis(fig[2, 1]; ylabel="slope", xlabel="iteration", aspect=1)
    hlines!(axr, [1.0]; color=:gray, linestyle=:dot)
    hlines!(axs, [1.0]; color=:gray, linestyle=:dot)
    nplotted = 0
    for (_, label) in ARCHS
        d = data[label].ab_paired
        if has(d)
            lines_and_points!(axr, d.iters, d.r; color=colors[label], label=label)
            lines_and_points!(axs, d.iters, d.slope; color=colors[label], label=label)
            nplotted += 1
        end
    end
    nplotted > 0 && axislegend(axr; position=:rb, framevisible=false)
    save(figpath("fig_headline_ab_check"), fig)
    return fig
end

function fig_firing()
    fig = Figure(size=(2 * CELL, CELL))
    for (col, (_, label)) in enumerate(ARCHS)
        d = data[label].firing_paired
        ax = Axis(fig[1, col]; title=label, xlabel="h_data — P(h=1 | data)", ylabel="h_model — P(h=1 | model)",
                  limits=(0, 1, 0, 1), aspect=1)
        lines!(ax, [0, 1], [0, 1]; color=:gray, linestyle=:dot)
        if has(d)
            scatter!(ax, last(d.data_rate), last(d.model_rate); color=colors[label])
        end
    end
    Label(fig[0, :], "Added-unit firing rate at last checkpoint (near y=x & away from 0/1 = healthy)"; fontsize=14)
    save(figpath("fig_firing_rate"), fig)
    return fig
end

function fig_norms()
    fig = Figure(size=(CELL, 2 * CELL))
    axw = Axis(fig[1, 1]; title="Weight norm (added block)", ylabel="mean ‖w‖", aspect=1)
    axg = Axis(fig[2, 1]; title="Gradient norm (added block)", ylabel="mean ‖∂w‖", xlabel="iteration", aspect=1)
    nplotted = 0
    for (_, label) in ARCHS
        wn = data[label].wn_paired
        gn = data[label].gn_paired
        has(wn) && (lines_and_points!(axw, wn.iters, wn.mean_norm; color=colors[label], label=label); nplotted += 1)
        has(gn) && lines_and_points!(axg, gn.iters, gn.mean_norm; color=colors[label], label=label)
    end
    nplotted > 0 && axislegend(axw; position=:rb, framevisible=false)
    save(figpath("fig_norms_added_block"), fig)
    return fig
end

fig_lpl()
fig_alignment("vh_check", (d, m) -> getfield(d, Symbol("vh_$m")), "VH alignment (hidden units)")
fig_alignment("v_check", (d, m) -> getfield(d, Symbol("v_$m")), "V alignment (single-site visible marginals ⟨v⟩)")
fig_ab_headline()
fig_firing()
fig_norms()

# =============================================================================
# WRITTEN REPORT
# =============================================================================
io = IOBuffer()
p(x...) = println(io, x...)

p("# Training analysis — N_ITERS=$(N_ITERS), PAIRED_ITERS=$(PAIRED_ITERS)")
p()
p("Figures saved to `$(REPORT_DIR)/`.")
p()

for (_, label) in ARCHS
    d = data[label]
    p("## $label")
    p()

    # --- 1. Did the individual RBMs train well? ---
    p("### 1. Individual RBM quality (A, B)")
    for m in (:A, :B)
        lpl = getfield(d, Symbol("lpl_$m"))
        val = getfield(d, Symbol("lplval_$m"))
        vh  = getfield(d, Symbol("vh_$m"))
        v   = getfield(d, Symbol("v_$m"))
        if !has(lpl)
            p("- RBM $m: no data yet.")
            continue
        end
        gap = has(val) ? tailmean(lpl.values) - tailmean(val.values) : NaN
        p("- RBM $m: final lpl=$(round(tailmean(lpl.values); digits=4)), lpl_val=$(has(val) ? round(tailmean(val.values); digits=4) : "n/a") (train-val gap=$(round(gap; digits=4)))")
        p("  vh_check r=$(round(tailmean(vh.r); digits=3)) slope=$(round(tailmean(vh.slope); digits=3)) — hidden units' data/model activations")
        p("  v_check r=$(round(tailmean(v.r); digits=3)) slope=$(round(tailmean(v.slope); digits=3)) — single-site visible marginals ⟨v⟩ (weak signal, sensitive to batch size)")
        if tailmean(v.r) < tailmean(vh.r) - 0.15
            p("  ⚠ v lags well behind vh: the true site-to-site variation in ⟨v⟩ is tiny, so this is easily swamped by finite-batch sampling noise — check against a longer/larger sample before reading this as a real model failure.")
        end
    end
    p()

    # --- 2. Are the added hidden units learning anything meaningful? ---
    p("### 2. Added hidden units (paired RBM)")
    freeze = d.freeze_paired
    vh = d.vh_paired
    wn = d.wn_paired
    gn = d.gn_paired
    fr = d.firing_paired
    if has(freeze)
        nviol = count(==( "VIOLATION"), freeze.status)
        p("- Freeze check: $(length(freeze.status) - nviol)/$(length(freeze.status)) OK" * (nviol > 0 ? " — **$nviol VIOLATIONS, freezing is broken!**" : " — private blocks stayed correctly frozen throughout."))
    end
    if has(vh)
        p("- vh_check: r=$(round(tailmean(vh.r); digits=3)), slope=$(round(tailmean(vh.slope); digits=3)), mean|Δ|=$(round(tailmean(vh.mean_abs_diff); digits=4))")
    end
    if has(wn) && has(gn)
        growth = wn.mean_norm[end] / wn.mean_norm[1]
        p("- Weight norm: $(round(wn.mean_norm[1]; digits=3)) → $(round(wn.mean_norm[end]; digits=3)) (×$(round(growth; digits=2)) growth)")
        p("- Gradient norm (final): mean=$(round(tailmean(gn.mean_norm); digits=3)) — " *
           (tailmean(gn.mean_norm) > 0.05 ? "still a live training signal." : "signal has largely died out (converged, or stuck)."))
    end
    if has(fr)
        dr, mr = last(fr.data_rate), last(fr.model_rate)
        ndead = count(x -> x < 0.05 || x > 0.95, dr)
        p("- Firing rate (last checkpoint): $(ndead)/$(length(dr)) added units pinned near 0 or 1 under data " *
           (ndead == 0 ? "(no dead units detected)." : "(possible dead units)."))
    end
    p()

    # --- 3. Are they learning interfamily correlations specifically? ---
    p("### 3. Interfamily (A-B) correlation — the headline question")
    ab = d.ab_paired
    if has(ab)
        p("- ab_check: r=$(round(tailmean(ab.r); digits=3)), slope=$(round(tailmean(ab.slope); digits=3)), mean|Δ|=$(round(tailmean(ab.mean_abs_diff); digits=4))")
        verdict = tailmean(ab.r) > 0.8 ? "learning real cross-family structure" :
                  tailmean(ab.r) > 0.4 ? "partial/early progress — needs more training or more H_ADD units to confirm" :
                  "not yet capturing cross-family correlations"
        p("  → **Verdict: $verdict** (r=$(round(tailmean(ab.r); digits=2)))")
    else
        p("- No ab_check data yet.")
    end
    p()
end

p("## Potts vs Binary — head-to-head")
p()
p("| metric | $potts_label | BinaryRBM |")
p("|---|---|---|")
function row(name, getter)
    vals = [has(getter(data[label])) ? round(tailmean(getter(data[label]).r); digits=3) : NaN for (_, label) in ARCHS]
    p("| $name (r) | $(vals[1]) | $(vals[2]) |")
end
row("vh_check (RBM A)", d -> d.vh_A)
row("v_check (RBM A)", d -> d.v_A)
row("ab_check (paired) — **headline**", d -> d.ab_paired)
row("v_check (paired)", d -> d.v_paired)
p()
p("Caveat: this is a **training-dynamics** report from logged summary statistics only —")
p("it does not yet verify the paired model against freshly *sampled* data (Gibbs/AIS),")
p("which is the only way to fully confirm the learned correlations generalize.")

report = String(take!(io))
open(joinpath(REPORT_DIR, "report.md"), "w") do f
    write(f, report)
end
println(report)
println("\nReport + figures written to $(REPORT_DIR)/")
