import { useState } from "react";
import { motion } from "framer-motion";
import StoryText from "./StoryText";

interface Props {
  template: string;
  answers: Record<number, string>;
  bertTop: Record<number, string | null>;
  onReplay: () => void;
  onNew: () => void;
}

export default function ResultCard({
  template,
  answers,
  bertTop,
  onReplay,
  onNew,
}: Props) {
  const [copied, setCopied] = useState(false);

  const indices = Object.keys(answers).map(Number);
  const surprises = indices.filter(
    (i) => answers[i].toLowerCase() !== (bertTop[i] ?? "").toLowerCase()
  ).length;
  const originality = indices.length
    ? Math.round((surprises / indices.length) * 100)
    : 0;

  const plainStory = template.replace(/\{\{(\d+)\}\}/g, (_, i) => answers[Number(i)]);

  const copy = () => {
    navigator.clipboard.writeText(plainStory).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <div className="card">
      <div className="label">Ton histoire</div>
      <StoryText
        template={template}
        renderSlot={(i) => (
          <motion.span
            className="reveal-word"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 * i + 0.2, duration: 0.4 }}
          >
            {answers[i]}
          </motion.span>
        )}
      />

      <div className="metrics">
        <div className="metric">
          <div className="k">Trous remplis</div>
          <div className="v">{indices.length}</div>
        </div>
        <div className="metric">
          <div className="k">Originalité</div>
          <div className="v">{originality}%</div>
        </div>
        <div className="metric">
          <div className="k">Surprises</div>
          <div className="v">{surprises}</div>
        </div>
      </div>
      <div className="bert-aside">
        « Originalité » = mots choisis différents de la 1ʳᵉ idée du mini-BERT.
      </div>

      <div className="comparison">
        <div className="label">Ton mot vs l'idée de TinyBERT</div>
        {indices.map((i) => {
          const mine = answers[i];
          const bert = bertTop[i];
          const same = !!bert && mine.toLowerCase() === bert.toLowerCase();
          return (
            <div className="cmp-row" key={i}>
              <span className="cmp-mine">{mine}</span>
              <span className="cmp-arrow">vs</span>
              <span className="cmp-bert">{bert ?? "—"}</span>
              <span className={`cmp-tag ${same ? "sync" : "orig"}`}>
                {same ? "sync" : "original"}
              </span>
            </div>
          );
        })}
      </div>

      <div className="actions">
        <button className="btn ghost" onClick={copy}>
          {copied ? "Copié !" : "Copier l'histoire"}
        </button>
        <button className="btn ghost" onClick={onReplay}>
          Rejouer ce début
        </button>
        <button className="btn primary" style={{ width: "auto" }} onClick={onNew}>
          Nouvelle histoire
        </button>
      </div>
    </div>
  );
}
