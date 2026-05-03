#!/usr/bin/env python3
"""
Quick CLI test runner for BiasScan pipeline.
Usage: python run_test.py [provider] [model]
"""

import asyncio
import json
import logging
import sys
import time
import os

# Load .env
from dotenv import load_dotenv
load_dotenv(override=True)

from backend.models.llm_client import LLMClient
from backend.orchestrator.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_test")

TEST_TEXT = """Study Selection and Characteristics
The literature search identified 247 records after removal of duplicates. Title and abstract screening excluded 189 records primarily due to irrelevance to multiple sclerosis or absence of microbiome-related outcomes. Fifty-eight full-text articles were assessed for eligibility, of which 46 were excluded because they lacked a healthy comparator, focused on non-human models, or did not report original microbiome data. Twelve observational studies met all inclusion criteria and were included in the final synthesis [1]–[12]. All included studies were published between 2019 and 2022 and collectively enrolled 1,126 individuals with multiple sclerosis and 1,019 healthy controls [1]–[12]. A PRISMA-style flow diagram describing the selection process is provided in Figure 1.
Across studies, designs were predominantly cross-sectional case–control, with two studies including longitudinal components [2], [6]. Seven studies focused primarily on relapsing–remitting MS [1], [4], [7], [9]–[11], one on treatment-naïve early MS [2], one on primary progressive MS [5], and three included mixed phenotypes with subgroup analyses [3], [6], [8]. Gut microbiome composition was assessed using 16S rRNA sequencing in ten studies [1]–[5], [7]–[10] and whole-metagenome shotgun sequencing in two studies [2], [6]. Four studies additionally integrated serum or fecal metabolomics [6], [10]–[12]. Geographic representation included Europe [2], [3], [5], the Middle East [6], East Asia [7], South America [1], and North Africa [4], enhancing external validity but also contributing to heterogeneity.
Risk of Bias Within Studies
Risk of bias was assessed using a modified Newcastle–Ottawa Scale for observational studies. Overall risk of bias was judged as moderate in eight studies [1], [3], [4], [7]–[11] and low in four studies [2], [5], [6], [12]. Selection bias was the most common concern, arising from single-center recruitment and incomplete matching of controls in several studies [1], [4], [7]. Performance bias related to diet, medication exposure, and antibiotic history was variably addressed; only five studies adjusted for disease-modifying therapies [1], [3], [6], [9], [10], and only three incorporated detailed dietary assessments [1], [2], [6]. Detection bias was limited by standardized sequencing pipelines, although variation in taxonomic databases and bioinformatic thresholds introduced some concern for comparability across studies [3], [7], [8]. Reporting bias was considered low, as all studies reported prespecified microbiome outcomes, although selective emphasis on statistically significant taxa was evident [4], [7], [11].
Gut Microbiome Composition in MS Versus Healthy Controls
All twelve studies reported significant differences in gut microbiome composition between individuals with MS and healthy controls [1]–[12], although the specific taxa implicated varied. Alpha diversity was reduced in MS in six studies [1], [2], [3], [7], [9], [10], unchanged in four studies [4], [6], [8], [11], and increased in two studies [5], [12]. When pooled descriptively, the direction of effect favored reduced microbial richness in MS, particularly in relapsing–remitting and treatment-naïve cohorts [1]–[3], [7], [9]. Studies reporting effect estimates showed standardized mean differences for Shannon diversity ranging from −0.32 to −0.71, with confidence intervals excluding the null in four studies [1], [2], [3], [7].
At the phylum level, consistent patterns included relative depletion of Firmicutes and enrichment of Bacteroidetes and Verrucomicrobia in MS [1], [4], [5], [7], [8], though effect sizes were modest and heterogeneous. At finer taxonomic resolution, recurrent findings included increased abundance of Akkermansia muciniphila [1], [4], [5], [7], [8], Bacteroides vulgatus [1], [4], Clostridium spp. [2], [7], and Desulfovibrio [5], alongside reduced abundance of Faecalibacterium prausnitzii [3], [6], Butyricicoccus [3], Prevotella [2], [4], and Bifidobacterium [1], [4]. Directionally concordant effects were observed in nine of twelve studies for Akkermansia muciniphila, with odds ratios for enrichment in MS ranging from 1.6 to 3.4 [1], [4], [5], [7], [8].
Functional and Metabolic Signatures
Four studies extended compositional analysis to functional inference or direct metabolomic measurement [6], [10]–[12]. Across these studies, MS was consistently associated with depletion of short-chain fatty acid–producing taxa and reduced abundance of genes involved in butyrate synthesis [6], [11], [12]. Two studies quantified fecal or inferred butyrate production, reporting lower mean relative abundance of butyrate-producing bacteria in MS (mean difference −2.7%, 95% CI −4.1 to −1.3) [6], [12]. Serum metabolomic analyses revealed reduced concentrations of indolelactate, indolepropionate, and other tryptophan-derived metabolites in MS, with large effect sizes (Cohen's d > 0.8) and narrow confidence intervals [6], [10], [12]. These metabolic alterations were strongly correlated with depletion of corresponding microbial producers, supporting biological plausibility [6].
Disease Phenotype, Activity, and Treatment Effects
Five studies examined associations between microbiome features and MS phenotype or disease activity [3], [5], [6], [9], [11]. Evidence for phenotype-specific signatures was mixed. One study focusing on primary progressive MS reported increased alpha diversity and enrichment of minor and rare taxa [5], contrasting with findings in relapsing–remitting MS [1], [3], [7], [9]. Three studies found inverse correlations between abundance of butyrate-producing bacteria and disability scores, with Spearman correlation coefficients ranging from −0.22 to −0.40 [3], [11], [12]. Treatment effects were inconsistently reported; however, two studies demonstrated that microbiome differences between MS and controls persisted after adjustment for disease-modifying therapies [1], [6], suggesting that dysbiosis is not solely treatment-driven.
Statistical Synthesis and Heterogeneity
Formal meta-analysis was limited by heterogeneity in sequencing platforms, taxonomic resolution, and outcome reporting [1]–[12]. Where comparable outcomes were available, random-effects synthesis demonstrated substantial heterogeneity, with I² values exceeding 70% for alpha-diversity measures [1]–[3], [7]. Meta-regression suggested that disease stage, treatment status, and geographic region accounted for a portion of this variability [3], [5], [6], though residual heterogeneity remained high. Sensitivity analyses excluding studies at higher risk of bias did not materially alter the direction of pooled estimates, supporting robustness of the primary findings [2], [5], [6].
Reporting Bias and Certainty of Evidence
Assessment of reporting bias was constrained by the small number of studies per outcome, precluding reliable funnel plot interpretation [1]–[12]. Nonetheless, selective reporting of significant taxa is likely [4], [7], [11]. Using a GRADE-informed framework adapted for observational microbiome research, certainty of evidence was rated as moderate for the presence of gut microbiome alterations in MS [1]–[12], low to moderate for specific taxonomic signatures [3], [7], and moderate for functional and metabolomic disruptions involving short-chain fatty acids and tryptophan metabolism [6], [10]–[12]. Confidence was downgraded primarily for inconsistency and indirectness but upgraded for biological coherence and reproducibility across independent cohorts [2], [6].
References
[1] Pellizoni, F. P. et al., "Detection of Dysbiosis and Increased Intestinal Permeability in Brazilian Patients with Relapsing–Remitting Multiple Sclerosis," Int. J. Environ. Res. Public Health, 2021.  [2] Ventura, R. E. et al., "Gut microbiome of treatment-naïve MS patients of different ethnicities early in disease course," Sci. Rep., 2019.  [3] Reynders, T. et al., "Gut microbiome variation is associated to Multiple Sclerosis phenotypic subtypes," Ann. Clin. Transl. Neurol., 2020.  [4] Mekky, J. et al., "Molecular characterization of the gut microbiome in Egyptian patients with relapsing remitting multiple sclerosis," Mult. Scler. Relat. Disord., 2022.  [5] Kozhieva, M. et al., "Primary progressive multiple sclerosis in a Russian cohort: relationship with gut bacterial diversity," BMC Microbiol., 2019.  [6] Levi, I. et al., "Potential role of indolelactate and butyrate in multiple sclerosis revealed by integrated microbiome–metabolome analysis," Cell Rep. Med., 2021.  [7] Alterations of the fecal microbiota in Chinese patients with multiple sclerosis, 2019.  [8] Gut microbiome variation is associated with multiple sclerosis, 2020.  [9] CXCR3⁺ T cells in multiple sclerosis correlate with reduced gut microbial diversity, 2020.  [10] Alterations in circulating fatty acids are associated with gut microbiota dysbiosis and inflammation in multiple sclerosis, 2021.  [11] Acetate correlates with disability in multiple sclerosis, 2020.  [12] Alterations of host–gut microbiome interactions in multiple sclerosis, 2019."""


async def run_pipeline(provider: str, model: str = None):
    """Run the BiasScan pipeline and print results."""
    logger.info(f"=== Testing BiasScan with {provider}/{model or 'default'} ===")
    start = time.time()

    try:
        llm = LLMClient(provider=provider, model=model)
        logger.info(f"LLMClient created: {llm.display_name}")
        logger.info(f"Sequential mode: {llm.rate_limiter.max_concurrent == 1}")

        pipeline = Pipeline(llm)

        async def on_progress(data):
            logger.info(f"  Phase: {data['phase']} (iteration {data['iteration']})")

        async def on_iteration(iteration):
            logger.info(
                f"  Iteration {iteration.iteration}: "
                f"composite={iteration.composite_score:.2f}, "
                f"ARGUS={iteration.argus_output.score:.1f}, "
                f"LIBRA={iteration.libra_output.score:.1f}, "
                f"LENS={iteration.lens_output.score:.1f}, "
                f"QUILL edits={len(iteration.quill_edits)}, "
                f"VIGIL={iteration.vigil_result.overall.value}"
            )
            if iteration.token_usage:
                total = iteration.token_usage.get("total", {})
                logger.info(
                    f"  Tokens: in={total.get('input_tokens', 0)}, "
                    f"out={total.get('output_tokens', 0)}, "
                    f"total={total.get('total_tokens', 0)}"
                )

        result = await pipeline.run(
            raw_text=TEST_TEXT,
            on_iteration=on_iteration,
            on_progress=on_progress,
            max_iterations=3,       # Keep it short for testing
            patience=2,
            threshold=0.05,
        )

        elapsed = time.time() - start
        logger.info(f"\n{'='*60}")
        logger.info(f"PIPELINE COMPLETE in {elapsed:.1f}s")
        logger.info(f"  Provider: {result.model_used}")
        logger.info(f"  Iterations: {len(result.iterations)}")
        logger.info(f"  Converged: {result.converged}")
        logger.info(f"  No bias detected: {result.no_bias_detected}")
        logger.info(f"  Final composite score: {result.final_composite:.2f}")
        logger.info(f"  Final sub-scores: {result.final_sub_scores}")

        if result.worst_finding:
            wf = result.worst_finding
            logger.info(f"  Worst finding: {wf.bias_type} ({wf.bias_name}) - {wf.severity.value}")
            logger.info(f"    Passage: {wf.passage[:100]}...")

        # Count all findings
        total_findings = 0
        for it in result.iterations:
            total_findings += len(it.argus_output.findings)
            total_findings += len(it.libra_output.findings)
            total_findings += len(it.lens_output.findings)
        logger.info(f"  Total findings across iterations: {total_findings}")

        # Show edits from last iteration
        if result.iterations and result.iterations[-1].quill_edits:
            edits = result.iterations[-1].quill_edits
            logger.info(f"  QUILL edits in final iteration: {len(edits)}")
            for e in edits[:3]:  # Show first 3
                logger.info(f"    [{e.operation}] {e.original[:60]}...")
                logger.info(f"      → {e.revised[:60]}...")

        # Save full result as JSON
        result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_result.json")
        result_dict = result.model_dump() if hasattr(result, 'model_dump') else result.dict()
        with open(result_path, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)
        logger.info(f"\n  Full results saved to: test_result.json")

        return result

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"PIPELINE FAILED after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "groq"
    model = sys.argv[2] if len(sys.argv) > 2 else None
    asyncio.run(run_pipeline(provider, model))
