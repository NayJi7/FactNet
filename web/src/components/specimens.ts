/**
 * 4 real posts with the actual outputs of the 5 content models, one picked at
 * random on load. The note under each is hand-written but its numbers are
 * computed from the values below.
 */
export interface Reading { model: string; p: number; primary?: boolean }

export interface Specimen {
  claim: string;
  readings: Reading[];
  /** what it shows */
  lesson: string;
}

export const SPECIMENS: Specimen[] = [
  {
    claim: "Vaccines cause autism, as thousands of parents have reported online.",
    readings: [
      { model: "RoBERTa (optimised)", p: 0.833 },
      { model: "BERT (fine-tuned)", p: 0.558 },
      { model: "RoBERTa (weight decay)", p: 0.476 },
      { model: "RoBERTa (fine-tuned)", p: 0.452, primary: true },
      { model: "TF-IDF + logistic regression", p: 0.428 },
    ],
    lesson: "A textbook antivaccine claim, and the most confident model calls it "
          + "reliable. Two others sit within a hair of the boundary. Nothing about "
          + "a single probability would have told you the five had split like this.",
  },
  {
    claim: "The unemployment rate has doubled since the new government took office, "
         + "according to sources online.",
    readings: [
      { model: "BERT (fine-tuned)", p: 0.638 },
      { model: "TF-IDF + logistic regression", p: 0.462 },
      { model: "RoBERTa (fine-tuned)", p: 0.232, primary: true },
      { model: "RoBERTa (weight decay)", p: 0.199 },
      { model: "RoBERTa (optimised)", p: 0.122 },
    ],
    lesson: "The widest disagreement of the four. One model reads this as reliable "
          + "and another as almost certainly false, on the same sentence, having "
          + "been trained on the same corpus under the same protocol.",
  },
  {
    claim: "Officials confirm the election was stolen through millions of "
         + "fraudulent mail-in ballots.",
    readings: [
      { model: "TF-IDF + logistic regression", p: 0.401 },
      { model: "RoBERTa (fine-tuned)", p: 0.349, primary: true },
      { model: "BERT (fine-tuned)", p: 0.346 },
      { model: "RoBERTa (weight decay)", p: 0.314 },
      { model: "RoBERTa (optimised)", p: 0.093 },
    ],
    lesson: "Here they agree, and all five land on the misleading side. Agreement "
          + "is possible, which is what makes the disagreement on the other "
          + "examples worth reporting rather than dismissing as noise.",
  },
  {
    claim: "Researchers at the university published a study showing the drug "
         + "reduced symptoms in 2019 trials.",
    readings: [
      { model: "RoBERTa (optimised)", p: 0.855 },
      { model: "RoBERTa (fine-tuned)", p: 0.811, primary: true },
      { model: "RoBERTa (weight decay)", p: 0.805 },
      { model: "BERT (fine-tuned)", p: 0.516 },
      { model: "TF-IDF + logistic regression", p: 0.493 },
    ],
    lesson: "The three transformers call an attributed study reliable and the two "
          + "simplest models cannot separate it from a coin. Which family you "
          + "trust decides the answer, and the benchmark does not settle that.",
  },
];

export const spread = (readings: Reading[]) =>
  Math.max(...readings.map((r) => r.p)) - Math.min(...readings.map((r) => r.p));
