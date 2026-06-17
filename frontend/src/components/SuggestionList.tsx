import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Blank, Candidate, fill } from "../api";

interface Props {
  blank: Blank;
  onChoose: (word: string, bertTop: string | null) => void;
}

export default function SuggestionList({ blank, onChoose }: Props) {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [loading, setLoading] = useState(true);
  const [custom, setCustom] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    setCustom("");
    fill(blank.masked_text, 5)
      .then((result) => active && setCandidates(result))
      .catch((e) => active && setError(String(e.message ?? e)))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [blank]);

  const bertTop = candidates[0]?.word ?? null;

  return (
    <div className="card">
      <div className="fill-head">
        <span className="dot" />
        TinyBERT propose pour le trou {blank.index + 1}
        <span className="badge bert" style={{ marginLeft: "auto" }}>
          {blank.hint}
        </span>
      </div>

      {loading && (
        <div className="label">
          <span className="spinner" />
          le mini-BERT réfléchit…
        </div>
      )}
      {error && <div className="error">{error}</div>}

      {!loading &&
        candidates.map((c) => (
          <div className="suggestion" key={c.word}>
            <button className="word" onClick={() => onChoose(c.word, bertTop)}>
              {c.word}
            </button>
            <div className="bar-track">
              <motion.div
                className="bar-fill"
                initial={{ width: 0 }}
                animate={{ width: `${Math.round(c.score * 100)}%` }}
                transition={{ duration: 0.5, ease: "easeOut" }}
              />
            </div>
            <span className="score-pct">{Math.round(c.score * 100)}%</span>
          </div>
        ))}

      <form
        className="custom"
        onSubmit={(e) => {
          e.preventDefault();
          const word = custom.trim();
          if (word) onChoose(word, bertTop);
        }}
      >
        <input
          placeholder="…ou écris ton propre mot"
          value={custom}
          onChange={(e) => setCustom(e.target.value)}
        />
        <button className="btn primary" style={{ width: "auto" }} type="submit">
          Valider
        </button>
      </form>
    </div>
  );
}
