Algorithm: Efficient and Exact Retrieval of Top N Papers by 18-Month Citation Count
Background

The goal is to identify the Top N papers ranked by citations received within the first 18 months after publication (18m citations).

Directly calculating 18m citations for every paper is computationally expensive because it requires retrieving all citing papers and filtering them by publication date.

We can significantly reduce the number of papers requiring this expensive calculation by leveraging the following property:

For any paper:

Total citations ≥ 18m citations

This means the total citation count serves as an upper bound for the 18m citation count. A paper whose total citations are lower than a known 18m citation threshold cannot possibly enter the final top N.

Algorithm

Assume we want to identify the Top N papers by 18m citations.

Step 1: Retrieve an Initial Candidate Set

Retrieve the top 2N papers ranked by total citations (total citations accumulated from publication date until today).

Using 2N instead of N generally produces a higher 18m citation threshold, which reduces the number of papers that need to be evaluated in later steps.

Let this initial set be:

S = top 2N papers by total citations
Step 2: Calculate 18m Citations for the Initial Candidates

For each paper in S:

Retrieve all papers that cite it.
For each citing paper, check whether its publication date is within 18 months after the target paper's publication date.
Count only those citations.

This gives the exact 18m citation count:

18m citations = Number of citing papers published within the first 18 months after the paper's publication date
Step 3: Determine the Minimum 18m Citation Threshold

Sort the papers in S by their 18m citation counts in descending order.

Take the 18m citation count of the N-th ranked paper:

m = 18m citation count of the N-th highest paper in S

m represents the minimum 18m citation count required to be in the current top N.

Step 4: Expand the Candidate Pool Using the Threshold

Retrieve all papers whose total citation count is greater than or equal to m:

T = all papers where total citations >= m

Any paper with:

total citations < m

can be safely excluded because:

18m citations ≤ total citations < m

and therefore it cannot surpass the current top N threshold.

Step 5: Calculate 18m Citations for Remaining Candidates

For every paper in T that is not already in S:

Retrieve its citing papers.
Count citations whose publication dates fall within the first 18 months after publication.

Combine these results with the 18m citation counts already calculated for papers in S.

Step 6: Generate the Final Leaderboard

Sort all papers in T by their 18m citation counts in descending order.

The final leaderboard is:

Top N papers by exact 18m citation count
Correctness Guarantee

This algorithm is mathematically exact and will never miss a true top N paper.

The reason is:

Total citations ≥ 18m citations

Therefore, any paper whose total citation count is lower than the current 18m threshold m cannot have enough 18m citations to enter the final top N.

Computational Advantage

Without this algorithm:

Calculate 18m citations for all papers in the field.

With this algorithm:

1. Calculate 18m citations for only 2N highly cited papers.
2. Use their 18m citation threshold to eliminate most papers.
3. Calculate 18m citations only for the remaining candidates.

In most scientific fields, the number of papers satisfying:

total citations >= m

will be much smaller than the total number of papers, reducing computation by orders of magnitude while preserving exact results.

Notes
The multiplier 2N is a practical default and can be adjusted (e.g., 3N, 5N, or 10N) depending on performance requirements.
A larger initial candidate set may produce a higher threshold m, resulting in a smaller final candidate pool T.
The algorithm is applicable to any fixed citation window, such as 6 months, 12 months, 18 months, or 24 months.

When this algorithm is used by the LLM-filtered PRI Stage 1 script, there are two independent multipliers:

1. LLM pool multiplier: how many exact top 18m-cited papers to send to the LLM relevance filter before returning the final top N on-topic papers.
2. Citation initial multiplier: the kN multiplier described in this algorithm, used internally to compute the exact top 18m-cited papers for the requested pool size.

For example, if the final target is 50 papers and the LLM pool multiplier is 1.5, Stage 1 asks this algorithm for the exact top 75 papers by 18m citations. The citation initial multiplier then controls this algorithm's initial set size S for that 75-paper request. In Stage 1, `--citation-initial-multiplier` defaults to 1.5 to preserve the existing pipeline behavior, while the standalone algorithm documentation above uses 2N as the general illustrative default.