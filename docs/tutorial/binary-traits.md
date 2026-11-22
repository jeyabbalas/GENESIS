# Binary traits

Case–control studies are analysed with the same commands as quantitative traits. Three
things change: the sample size that must be supplied, the scale on which heritability is
expressed, and the summaries that are of interest — sibling relative risk and the
discriminatory accuracy of a polygenic risk score.

:::{admonition} The example in this chapter is simulated
:class: important

No public case–control data set is distributed with the software, so this chapter uses
summary statistics **simulated from a known effect-size distribution** with the real LD
of the European reference. They are not from any real disease. The benefit is that the
estimates can be compared with the truth.
:::

## What is different

**Sample size.** Supply the effective sample size
$n_\text{eff} = n_\text{case}n_\text{control}/(n_\text{case}+n_\text{control})$, which the
software computes itself when the file has case and control counts
({ref}`tut-effective-n`).

**Scale.** With that sample size, $\hat\beta_k = z_k/\sqrt{n_{\text{eff},k}}$ is a log odds
ratio per standard deviation of genotype. Consequently

- $\sigma_h^2$ is the variance of log odds ratios (per standard deviation of genotype)
  among susceptibility SNPs in cluster $h$;
- $h^2 = M\pi_c\sum_h p_h\sigma_h^2$ is the variance of the polygenic log odds across
  individuals — "heritability on the log-odds scale". It is not comparable with the
  liability-scale heritability of other methods until converted
  ({ref}`tut-h2-scales`).

## A simulated case–control GWAS

The script draws joint effects from a three-component model for the 1,070,777 reference
SNPs, turns them into marginal effects through the LD of the reference, adds
LD-correlated sampling noise, and writes the file a consortium would distribute: per-allele
log odds ratios with standard errors, allele frequencies and per-SNP case and control
counts. The design is 120,000 cases and 100,000 controls, with 90–100 % of the samples
available at each SNP.

```{literalinclude} ../_scripts/examples/simulate_disease.py
:language: python
```

```{literalinclude} ../_scripts/output/simulate_disease.txt
:language: console
```

(tut-check-n)=
### First, check the sample-size convention

Before fitting a case–control file, confirm what its sample-size column means. The
standard error of a per-allele log odds ratio determines the effective sample size,
$n_\text{eff} \approx 1/\{2f(1-f)\,\mathrm{SE}^2\}$, so the file can be tested against
itself:

```{literalinclude} ../_scripts/examples/check_effective_n.py
:language: python
```

```{literalinclude} ../_scripts/output/check_effective_n.txt
:language: console
```

The ratio is 1 for the convention GENESIS needs and 4 for a METAL-style effective N; the
total sample size gives 4 or more (exactly 4 when cases and controls are equally many).
In real data the ratio is not exactly 1 — imputation uncertainty and covariate adjustment
change standard errors by a few per cent — but it is never close to 4 by accident.

### Fit

The file has case and control columns, so no sample-size option is needed. The two
prevalences are optional and only add the heritability on other scales to the report:

```{literalinclude} ../_scripts/output/fit_disease.txt
:language: console
:end-before: Fitting the two-component model
```

```{literalinclude} ../_scripts/output/fit_disease.txt
:language: text
:start-at: GENESIS fit: M3
```

### Compare with the truth

```{literalinclude} ../_scripts/examples/disease.py
:language: python
```

```{literalinclude} ../_scripts/output/disease.txt
:language: console
```

Read the first two columns first.

- **Total heritability** is recovered within its standard error.
- **The number of susceptibility SNPs is too low**, by about one standard error — the
  direction that the paper's simulations lead one to expect ({doc}`concepts`). The
  smallest simulated effects cannot be told from zero at this sample size.
- **The split between clusters is the least accurate part.** The fit attributes less
  heritability to the cluster of larger effects, and more to the other, than was
  simulated.
- **The distribution itself is recovered much better than its components** (second
  block). The numbers of SNPs with effects above 0.005, 0.01 and 0.02 are within about a
  quarter of the truth. This is the paper's observation: component-specific parameters
  are biased, in directions that compensate, while the effect-size distribution is stable
  in the range where the study has power. Report distributions and tail counts in
  preference to cluster parameters.

The simulation is deliberately not a draw from the fitting model: marginal effects are
generated from the actual pairwise LD of the reference, whereas the model summarises each
neighbourhood by its size and LD score. This is the situation of real data.

(tut-chisq-filter)=
### What the $z^2 > 80$ filter does

The last column is the same fit without the quality-control filter that removes markers
with $z^2 > 80$ (`--max-chisq inf`); it analyses a few hundred more markers. The total
heritability stays within about one standard error of the truth, and the tail counts
barely move. The cluster parameters do: the estimated number of SNPs in the cluster of
larger effects halves. In this simulation the default, filtered fit is the closer one in
the first block, even though each removed marker is a genuine draw from the simulated
model.

A likely reason — an interpretation, not something shown here — is that the model
represents the LD between a marker and the susceptibility SNPs it tags by one average
$r^2$ per neighbourhood. That approximation is poorest for the few markers in
near-perfect LD with a large-effect SNP, which are the markers with the largest
statistics.

Practical consequences:

- Keep the default filter, as the paper does, and report how many markers it removed (the
  QC table has the number).
- Treat cluster-specific quantities as less robust than the total heritability and the
  overall shape of the distribution; if they matter, fit both ways and compare.
- In real data the strongest loci are often outliers to any normal mixture. With the
  filter on they are simply not part of the description: heritability and projections
  then refer to the rest of the genome.

(tut-h2-scales)=
## Heritability on other scales

The log shows the heritability on two further scales because the fit was given the
lifetime risk in the population, $K$ (`--population-prevalence`), and the proportion of
cases in the study, $P$ (`--sample-prevalence`):

$$
h^2_\text{observed} = h^2_\text{log-odds}\; P(1-P),
\qquad
h^2_\text{liability} = h^2_\text{observed}\;\frac{K^2(1-K)^2}{\varphi(t)^2\,P(1-P)},
$$

where $t = \Phi^{-1}(1-K)$ is the liability threshold and $\varphi$ the standard normal
density ([Lee et al., 2011](../reference/bibliography.md#ref-lee2011)). The first step
uses the fact that, for small effects, a log odds ratio $b$ corresponds to an effect
$bP(1-P)$ on the 0/1 scale; the second is the standard conversion from the observed scale
in an ascertained sample to the liability scale.

In Python, pass `population_prevalence=` and `sample_prevalence=` to `gg.fit`, or convert
afterwards with {func}`genesis_gwas.h2_transfer` as in the script above. The liability
scale is the one to use when comparing with LD-score regression or with family studies.
Its value depends on $K$, which is rarely known precisely; quote the $K$ you used.

## Sibling relative risk

If the polygenic log odds $G$ is normally distributed with variance $h^2$ and the disease
is not too common, the risk to a sibling of a case, relative to the population risk, is

$$
\lambda_S = \exp\bigl(\operatorname{cov}(G_\text{sib 1}, G_\text{sib 2})\bigr)
          = \exp\bigl(h^2/2\bigr),
$$

because siblings share half of the additive genetic variance. This is the "sibling
relative risk" of the output: the familial aggregation that the common SNPs of the
reference panel would produce on their own. Comparing $\log\lambda_S$ with the logarithm
of the familial relative risk observed in epidemiological studies gives the fraction of
familial risk that common variants explain.

## Risk prediction

For a disease, {func}`genesis_gwas.polyrisk_predict` reports

`auc`
: the expected area under the ROC curve of the polygenic risk score,
  $\Phi(\mu/\sqrt2)$, where $\mu$ is the expected difference between the mean score of
  cases and of controls, in standard deviations of the score
  ([Chatterjee et al., 2013](../reference/bibliography.md#ref-chatterjee2013));

`ppi`
: for each risk multiple $k$ in `k_fold`, the proportion of the population whose
  predicted risk is at least $k$ times the average;

`pci`
: the proportion of future cases who are in that group.

The sample size is again the *effective* one. The last table of the output above gives
these numbers for scores built from GWAS of increasing size, and
{numref}`fig-auc-disease` shows the whole curve.

```{figure} ../_static/figures/auc-disease.svg
:name: fig-auc-disease
:alt: Left, the expected AUC of a polygenic risk score for the simulated disease against the effective GWAS sample size, for the optimal threshold and for genome-wide significant SNPs, approaching a ceiling. Right, the percentage of the population at twice the average risk or more and the percentage of future cases in that group.

**Expected performance of a polygenic risk score for the simulated disease**, from the
fitted model. (a) AUC with the optimal P-value threshold and with genome-wide significant
SNPs only; the dashed line is the ceiling set by the heritability. (b) The proportion of
the population predicted to be at twice the average risk or more, and the proportion of
future cases in that group.
```

The AUC approaches its ceiling early; the size of the high-risk group, and the share of
cases it contains, keep growing for longer. The paper makes the same observation for real
diseases: beyond a sample size of about 100,000 the AUC rises only modestly, while the
ability to identify people at the extremes of risk still improves.

From the command line:

```{literalinclude} ../_scripts/output/cli_predict.txt
:language: console
```

`genesis predict` takes the number of independent SNPs to be the number of reference
SNPs, where the script above used the paper's 200,000. More null SNPs compete for
selection, so the optimal threshold is stricter and the AUC slightly lower.

:::{admonition} Differences from the original software
:class: note

`ppi` and `pci` use a standard deviation of the predicted log odds equal to $\mu$, which
is consistent with the reported AUC. The original function uses $\sqrt\mu$, which
overstates the spread of risk; `polyrisk_predict(..., legacy=True)` reproduces it. See
[Differences from the original](../reference/differences.md).
:::

## Checklist for a binary trait

- [ ] The sample size is $n_\text{case}n_\text{control}/(n_\text{case}+n_\text{control})$ —
      verified with the standard-error check above when allele frequencies are available.
- [ ] Effects are log odds ratios (the software takes the logarithm of an `OR` column
      itself).
- [ ] You know how many markers the $z^2 > 80$ filter removed.
- [ ] Heritability is reported with its scale; for the liability scale, with the $K$
      used.
- [ ] Sample sizes passed to `project` and `polyrisk_predict` are effective sample sizes.
