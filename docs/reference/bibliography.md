# References

Every publication cited in this documentation, grouped by topic, with one line on why
it matters here. Entries can be linked from any page as
`[Zhang et al. 2018](#ref-zhang2018)`.

:::{tip}
If you use this software, please cite the method paper,
[Zhang et al. (2018)](#ref-zhang2018), and the software itself (see
[How to cite](../index.md#how-to-cite)).
:::

## The method and its applications

(ref-zhang2018)=
**Zhang et al. (2018).** Zhang Y, Qi G, Park J-H, Chatterjee N. Estimation of complex
effect-size distributions using summary-level statistics from genome-wide association
studies across 32 complex traits. *Nature Genetics* **50**, 1318–1326.
[doi:10.1038/s41588-018-0193-x](https://doi.org/10.1038/s41588-018-0193-x)\
→ The paper that introduces GENESIS: the mixture model for joint effects, the composite
likelihood for summary statistics, the sandwich variance, the modified BIC, and the
projections. This package implements it.

(ref-park2010)=
**Park et al. (2010).** Park J-H, Wacholder S, Gail MH, Peters U, Jacobs KB, Chanock SJ,
Chatterjee N. Estimation of effect size distribution from genome-wide association
studies and implications for future discoveries. *Nature Genetics* **42**, 570–575.
[doi:10.1038/ng.610](https://doi.org/10.1038/ng.610)\
→ The precursor: effect-size distributions estimated from reported discoveries and study
power, and the idea of projecting the yield of larger studies.

(ref-chatterjee2013)=
**Chatterjee et al. (2013).** Chatterjee N, Wheeler B, Sampson J, Hartge P, Chanock SJ,
Park J-H. Projecting the performance of risk prediction based on polygenic analyses of
genome-wide association studies. *Nature Genetics* **45**, 400–405.
[doi:10.1038/ng.2579](https://doi.org/10.1038/ng.2579)\
→ The theory behind `polyrisk_predict`: expected predictive performance of a polygenic
score as a function of sample size, effect-size distribution and selection threshold.

(ref-zhang2020b)=
**Zhang et al. (2020b).** Zhang YD, Hurson AN, Zhang H, Choudhury PP, Easton DF, Milne RL,
*et al.* Assessment of polygenic architecture and risk prediction based on common
variants across fourteen cancers. *Nature Communications* **11**, 3353.
[doi:10.1038/s41467-020-16483-3](https://doi.org/10.1038/s41467-020-16483-3)\
→ Applies the method of Zhang et al. (2018) to GWAS summary statistics of fourteen
cancers, including breast cancer; a template for cancer applications.

(ref-zhang2020)=
**Zhang et al. (2020).** Zhang H, Ahearn TU, Lecarpentier J, Barnes D, Beesley J, Qi G,
*et al.* Genome-wide association study identifies 32 novel breast cancer susceptibility
loci from overall and subtype-specific analyses. *Nature Genetics* **52**, 572–581.
[doi:10.1038/s41588-020-0609-2](https://doi.org/10.1038/s41588-020-0609-2)\
→ The European-ancestry breast cancer GWAS (overall and by subtype) whose kind of
summary statistics the Confluence project extends to more ancestries and larger
samples.

(ref-langoallen2010)=
**Lango Allen et al. (2010).** Lango Allen H, Estrada K, Lettre G, Berndt SI, Weedon MN,
Rivadeneira F, *et al.* Hundreds of variants clustered in genomic loci and biological
pathways affect human height. *Nature* **467**, 832–838.
[doi:10.1038/nature09410](https://doi.org/10.1038/nature09410)\
→ Source of the GIANT height summary statistics used as the worked example of the
original package and throughout this documentation.

(ref-yengo2018)=
**Yengo et al. (2018).** Yengo L, Sidorenko J, Kemper KE, Zheng Z, Wood AR, Weedon MN,
*et al.*, the GIANT Consortium. Meta-analysis of genome-wide association studies for
height and body mass index in ∼700000 individuals of European ancestry. *Human
Molecular Genetics* **27**, 3641–3649.
[doi:10.1093/hmg/ddy271](https://doi.org/10.1093/hmg/ddy271)\
→ A later GWAS of height in about 700,000 individuals; used in the tutorial to compare
a projection made from the 2010 data with what a larger study found.

% Additional entries for this section go here.

## LD scores, reference panels and summary statistics

(ref-bulik-sullivan2015)=
**Bulik-Sullivan et al. (2015).** Bulik-Sullivan BK, Loh P-R, Finucane HK, Ripke S, Yang J,
Schizophrenia Working Group of the Psychiatric Genomics Consortium, Patterson N, Daly MJ,
Price AL, Neale BM. LD Score regression distinguishes confounding from polygenicity in
genome-wide association studies. *Nature Genetics* **47**, 291–295.
[doi:10.1038/ng.3211](https://doi.org/10.1038/ng.3211)\
→ LD scores, the bias correction applied to $r^2$, and the regression that GENESIS uses
for starting values; the inflation term $a$ plays the role of the LDSC intercept.

(ref-zheng2017)=
**Zheng et al. (2017).** Zheng J, Erzurumluoglu AM, Elsworth BL, Kemp JP, Howe L,
Haycock PC, *et al.* LD Hub: a centralized database and web interface to perform LD
score regression that maximizes the potential of summary level GWAS data for SNP
heritability and genetic correlation analysis. *Bioinformatics* **33**, 272–279.
[doi:10.1093/bioinformatics/btw613](https://doi.org/10.1093/bioinformatics/btw613)\
→ Origin of the quality-control filters (sample-size, MHC, $\chi^2 > 80$) that the paper
and this package apply to summary statistics.

(ref-yang2012)=
**Yang et al. (2012).** Yang J, Ferreira T, Morris AP, Medland SE, Genetic Investigation of
ANthropometric Traits (GIANT) Consortium, DIAGRAM Consortium, *et al.* Conditional and
joint multiple-SNP analysis of GWAS summary statistics identifies additional variants
influencing complex traits. *Nature Genetics* **44**, 369–375.
[doi:10.1038/ng.2213](https://doi.org/10.1038/ng.2213)\
→ The relation between marginal and joint effects through LD,
$\beta_k = \sum_m \beta^{(J)}_m \rho_{km}$, on which the model is built.

(ref-auton2015)=
**1000 Genomes Project Consortium (2015).** Auton A, Abecasis GR, Altshuler DM, Durbin RM,
*et al.* A global reference for human genetic variation. *Nature* **526**, 68–74.
[doi:10.1038/nature15393](https://doi.org/10.1038/nature15393)\
→ The Phase 3 genotypes from which the European (489 individuals), East Asian (504)
and African-ancestry (661) LD references were computed, and the natural source of
panels for other ancestries.

(ref-altshuler2010)=
**International HapMap 3 Consortium (2010).** Integrating common and rare genetic variation
in diverse human populations. *Nature* **467**, 52–58.
[doi:10.1038/nature09298](https://doi.org/10.1038/nature09298)\
→ Defines the HapMap3 SNP set; the reference panel is the 1.07 million HapMap3 SNPs with
minor-allele frequency of at least 5%.

(ref-chang2015)=
**Chang et al. (2015).** Chang CC, Chow CC, Tellier LCAM, Vattikuti S, Purcell SM, Lee JJ.
Second-generation PLINK: rising to the challenge of larger and richer datasets.
*GigaScience* **4**, 7.
[doi:10.1186/s13742-015-0047-8](https://doi.org/10.1186/s13742-015-0047-8)\
→ PLINK 1.9, whose binary genotype format `genesis ld build` reads and whose printed
$r^2$ values the original LD tables were derived from.

% Additional entries for this section go here.

## Composite likelihood, sandwich variance and model selection

(ref-lindsay1988)=
**Lindsay (1988).** Lindsay BG. Composite likelihood methods. In *Statistical Inference
from Stochastic Processes*, Contemporary Mathematics **80**, 221–239. American
Mathematical Society.
[doi:10.1090/conm/080/999014](https://doi.org/10.1090/conm/080/999014)\
→ Foundational paper: multiplying valid marginal likelihoods gives consistent
estimating equations even when the dependence between components is ignored.

(ref-varin2011)=
**Varin, Reid & Firth (2011).** Varin C, Reid N, Firth D. An overview of composite
likelihood methods. *Statistica Sinica* **21**, 5–42.
[Journal page](https://www3.stat.sinica.edu.tw/statistica/J21N1/J21N11/J21N11.html)\
→ Review covering the Godambe (sandwich) information and composite-likelihood
information criteria used here.

(ref-godambe1960)=
**Godambe (1960).** Godambe VP. An optimum property of regular maximum likelihood
estimation. *Annals of Mathematical Statistics* **31**, 1208–1211.
[doi:10.1214/aoms/1177705693](https://doi.org/10.1214/aoms/1177705693)\
→ Estimating-function theory; the "Godambe information" $I J^{-1} I$ is the inverse of
the sandwich covariance.

(ref-white1982)=
**White (1982).** White H. Maximum likelihood estimation of misspecified models.
*Econometrica* **50**, 1–25.
[doi:10.2307/1912526](https://doi.org/10.2307/1912526)\
→ Why the sandwich form $I^{-1} J I^{-1}$ gives valid standard errors when the working
likelihood is not the true joint likelihood.

(ref-heagerty2000)=
**Heagerty & Lumley (2000).** Heagerty PJ, Lumley T. Window subsampling of estimating
functions with application to regression models. *Journal of the American Statistical
Association* **95**, 197–211.
[doi:10.1080/01621459.2000.10473914](https://doi.org/10.1080/01621459.2000.10473914)\
→ Variance estimation for estimating functions with local ("banded") dependence — the
basis of summing scores over tagging neighbourhoods.

(ref-lumley1999)=
**Lumley & Heagerty (1999).** Lumley T, Heagerty P. Weighted empirical adaptive variance
estimators for correlated data regression. *Journal of the Royal Statistical Society,
Series B* **61**, 459–477.
[doi:10.1111/1467-9868.00187](https://doi.org/10.1111/1467-9868.00187)\
→ Companion to the previous entry; empirical covariance of scores within windows of
correlated observations.

(ref-gao2010)=
**Gao & Song (2010).** Gao X, Song PX-K. Composite likelihood Bayesian information
criteria for model selection in high-dimensional data. *Journal of the American
Statistical Association* **105**, 1531–1540.
[doi:10.1198/jasa.2010.tm09414](https://doi.org/10.1198/jasa.2010.tm09414)\
→ BIC for composite likelihoods with the effective number of parameters
$d_s = \operatorname{tr}(I^{-1} J)$.

(ref-chen2008)=
**Chen & Chen (2008).** Chen J, Chen Z. Extended Bayesian information criteria for model
selection with large model spaces. *Biometrika* **95**, 759–771.
[doi:10.1093/biomet/asn034](https://doi.org/10.1093/biomet/asn034)\
→ The extended BIC with tuning constant $\gamma$, the origin of the extra
$2\gamma\,d_s\log 5$ term in the model-selection criterion.

% Additional entries for this section go here.

## The EM algorithm

(ref-dempster1977)=
**Dempster, Laird & Rubin (1977).** Dempster AP, Laird NM, Rubin DB. Maximum likelihood
from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society,
Series B* **39**, 1–38 (with discussion).
[doi:10.1111/j.2517-6161.1977.tb01600.x](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x)\
→ The EM algorithm and its ascent property, here applied with the numbers of tagged
susceptibility SNPs as the missing data.

(ref-wu1983)=
**Wu (1983).** Wu CFJ. On the convergence properties of the EM algorithm. *Annals of
Statistics* **11**, 95–103.
[doi:10.1214/aos/1176346060](https://doi.org/10.1214/aos/1176346060)\
→ Convergence of EM and generalised EM to stationary points; the reason a partial
M-step (a few Newton steps) is still a valid algorithm, and why convergence is only
linear.

(ref-louis1982)=
**Louis (1982).** Louis TA. Finding the observed information matrix when using the EM
algorithm. *Journal of the Royal Statistical Society, Series B* **44**, 226–233.
[doi:10.1111/j.2517-6161.1982.tb01203.x](https://doi.org/10.1111/j.2517-6161.1982.tb01203.x)\
→ The identity that expresses the observed information of a mixture through
complete-data quantities — the structure of the information matrix computed in
`core.likelihood`.

% Additional entries for this section go here.

## Heritability scales, risk prediction and false discovery rates

(ref-lee2011)=
**Lee et al. (2011).** Lee SH, Wray NR, Goddard ME, Visscher PM. Estimating missing
heritability for disease from genome-wide association studies. *American Journal of
Human Genetics* **88**, 294–305.
[doi:10.1016/j.ajhg.2011.02.002](https://doi.org/10.1016/j.ajhg.2011.02.002)\
→ Conversion between the observed and liability scales, implemented in `h2_transfer`.

(ref-wray2007)=
**Wray, Goddard & Visscher (2007).** Wray NR, Goddard ME, Visscher PM. Prediction of
individual genetic risk to disease from genome-wide association studies. *Genome
Research* **17**, 1520–1528.
[doi:10.1101/gr.6665407](https://doi.org/10.1101/gr.6665407)\
→ Relation between variance explained by a genetic profile and the area under the ROC
curve.

(ref-dudbridge2013)=
**Dudbridge (2013).** Dudbridge F. Power and predictive accuracy of polygenic risk scores.
*PLoS Genetics* **9**, e1003348.
[doi:10.1371/journal.pgen.1003348](https://doi.org/10.1371/journal.pgen.1003348)\
→ Alternative derivation of polygenic-score accuracy as a function of sample size and
genetic architecture.

(ref-efron2002)=
**Efron & Tibshirani (2002).** Efron B, Tibshirani R. Empirical Bayes methods and false
discovery rates for microarrays. *Genetic Epidemiology* **23**, 70–86.
[doi:10.1002/gepi.1124](https://doi.org/10.1002/gepi.1124)\
→ The local false-discovery rate, the quantity returned as `p_null` by
`posterior_marginal`.

(ref-stephens2017)=
**Stephens (2017).** Stephens M. False discovery rates: a new deal. *Biostatistics* **18**,
275–294.
[doi:10.1093/biostatistics/kxw041](https://doi.org/10.1093/biostatistics/kxw041)\
→ Empirical-Bayes shrinkage with a unimodal prior; context for the posterior-mean
effect sizes.

% Additional entries for this section go here.

## Related models of genetic architecture

(ref-moser2015)=
**Moser et al. (2015).** Moser G, Lee SH, Hayes BJ, Goddard ME, Wray NR, Visscher PM.
Simultaneous discovery, estimation and prediction analysis of complex traits using a
Bayesian mixture model. *PLoS Genetics* **11**, e1004969.
[doi:10.1371/journal.pgen.1004969](https://doi.org/10.1371/journal.pgen.1004969)\
→ BayesR, the individual-level normal-mixture model against which the paper compares
GENESIS.

(ref-holland2020)=
**Holland et al. (2020).** Holland D, Frei O, Desikan R, Fan C-C, Shadrin AA, Smeland OB,
Sundar VS, Thompson P, Andreassen OA, Dale AM. Beyond SNP heritability: polygenicity and
discoverability of phenotypes estimated with a univariate Gaussian mixture model.
*PLoS Genetics* **16**, e1008612.
[doi:10.1371/journal.pgen.1008612](https://doi.org/10.1371/journal.pgen.1008612)\
→ MiXeR, a contemporaneous summary-statistics mixture model with a different treatment
of LD (cited by the paper as a 2017 preprint).

(ref-gazal2017)=
**Gazal et al. (2017).** Gazal S, Finucane HK, Furlotte NA, Loh P-R, Palamara PF, Liu X,
Schoech A, Bulik-Sullivan B, Neale BM, Gusev A, Price AL. Linkage
disequilibrium–dependent architecture of human complex traits shows action of negative
selection. *Nature Genetics* **49**, 1421–1427.
[doi:10.1038/ng.3954](https://doi.org/10.1038/ng.3954)\
→ Evidence that effect sizes depend on local LD — one of the model assumptions
(independence of effects and LD) examined in the paper's sensitivity analyses.

(ref-speed2017)=
**Speed et al. (2017).** Speed D, Cai N, the UCLEB Consortium, Johnson MR, Nejentsev S,
Balding DJ. Reevaluation of SNP heritability in complex human traits. *Nature Genetics*
**49**, 986–992.
[doi:10.1038/ng.3865](https://doi.org/10.1038/ng.3865)\
→ Shows how heritability estimates depend on assumptions about allele frequency and LD;
context for the limitations discussed in the paper.

% Additional entries for this section go here.

## Software

(ref-harris2020)=
**Harris et al. (2020).** Harris CR, Millman KJ, van der Walt SJ, Gommers R, Virtanen P,
Cournapeau D, *et al.* Array programming with NumPy. *Nature* **585**, 357–362.
[doi:10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)\
→ The CPU backend.

(ref-virtanen2020)=
**Virtanen et al. (2020).** Virtanen P, Gommers R, Oliphant TE, Haberland M, Reddy T,
Cournapeau D, *et al.* SciPy 1.0: fundamental algorithms for scientific computing in
Python. *Nature Methods* **17**, 261–272.
[doi:10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)\
→ Special functions, distributions and optimisation used by the downstream analyses.

(ref-paszke2019)=
**Paszke et al. (2019).** Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chanan G,
*et al.* PyTorch: an imperative style, high-performance deep learning library.
*Advances in Neural Information Processing Systems* **32**, 8024–8035.
[Proceedings](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html)\
→ The GPU backend.

% Additional entries for this section go here.

## Software and data resources

- Original GENESIS R package: <https://github.com/yandorazhang/GENESIS> (commit
  `e4e6894`, the version this implementation was validated against).
- Original GENESISasian R package: <https://github.com/yandorazhang/GENESISasian>
  (commit `4df90ac`), the source of the East Asian reference.
- 1000 Genomes Project Phase 3 integrated call set v5a (release 20130502), VCF files
  and sample panel, from which the African-ancestry reference and the additional LD
  configurations of the East Asian reference were computed:
  <https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/> (mirrored in the AWS
  Open Data registry, <https://registry.opendata.aws/1000-genomes/>).
- LDSC 1000 Genomes Phase 3 PLINK files (489 European samples) used to validate the LD
  builder: S-LDSC reference files, Zenodo record
  [10515792](https://zenodo.org/records/10515792).
- UCSC liftOver chain `hg19ToHg38.over.chain.gz`, used for the GRCh38 positions of the
  downloadable references: <https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/>.
- Confluence project (NCI Division of Cancer Epidemiology and Genetics):
  <https://dceg.cancer.gov/research/cancer-types/breast-cancer/confluence-project>.
- NIH HPC (Biowulf) user documentation: <https://hpc.nih.gov/>.
