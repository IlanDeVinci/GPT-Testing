import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  ModelInfo,
  Story,
  fetchModels,
  fetchOpenings,
  generate,
  setModel,
} from "./api";
import StoryText from "./components/StoryText";
import SuggestionList from "./components/SuggestionList";
import ResultCard from "./components/ResultCard";

type Phase = "start" | "loading" | "fill" | "reveal";

export default function App() {
  const [phase, setPhase] = useState<Phase>("start");
  const [openings, setOpenings] = useState<string[]>([]);
  const [opening, setOpening] = useState("");
  const [nBlanks, setNBlanks] = useState(4);
  const [story, setStory] = useState<Story | null>(null);
  const [current, setCurrent] = useState(0);
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [bertTop, setBertTop] = useState<Record<number, string | null>>({});
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [activeModel, setActiveModel] = useState<string>("");
  const [activeModelLabel, setActiveModelLabel] = useState<string>("");
  const [activeCheckpoint, setActiveCheckpoint] = useState<string | null>(null);
  const [switching, setSwitching] = useState(false);

  useEffect(() => {
    fetchOpenings()
      .then((list) => {
        setOpenings(list);
        setOpening((prev) => prev || list[0] || "");
      })
      .catch(() =>
        setError("API injoignable — lance le backend sur le port 8000."),
      );
    fetchModels()
      .then((state) => {
        setModels(state.models);
        setActiveModel(state.active);
        setActiveModelLabel(state.active_label);
        setActiveCheckpoint(state.active_checkpoint);
      })
      .catch(() => {});
  }, []);

  async function changeModel(lineage: string) {
    if (lineage === activeModel || switching) return;
    setSwitching(true);
    setError(null);
    try {
      const result = await setModel(lineage);
      setActiveModel(result.active);
      setActiveModelLabel(result.active_label);
      setActiveCheckpoint(result.active_checkpoint);
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setSwitching(false);
    }
  }

  async function start(sameStory: boolean) {
    setPhase("loading");
    setError(null);
    try {
      const result = await generate(
        sameStory && story ? story.opening : opening,
        nBlanks,
      );
      setStory(result);
      setAnswers({});
      setBertTop({});
      setCurrent(0);
      setPhase("fill");
    } catch (e) {
      setError(String((e as Error).message ?? e));
      setPhase("start");
    }
  }

  function choose(word: string, top: string | null) {
    if (!story) return;
    const blank = story.blanks[current];
    setAnswers((a) => ({ ...a, [blank.index]: word }));
    setBertTop((b) => ({ ...b, [blank.index]: top }));
    if (current + 1 < story.blanks.length) {
      setCurrent(current + 1);
    } else {
      setPhase("reveal");
    }
  }

  return (
    <div className="app">
      <header className="brand">
        <div>
          <h1>Histoires à Trous</h1>
          <p>Le mini-GPT invente, ton mini-BERT souffle les mots.</p>
        </div>
        <div className="badges">
          <span className="badge gpt">GPT</span>
          <span className="badge bert">TinyBERT</span>
        </div>
      </header>

      <div className="model-bar">
        <div className="model-copy">
          <span className="model-label">Modèle GPT actif</span>
          <strong>{activeModelLabel || "Chargement…"}</strong>
          {activeCheckpoint && (
            <span className="model-checkpoint">{activeCheckpoint}</span>
          )}
        </div>
        <div className="model-controls">
          <select
            value={activeModel}
            disabled={switching || models.length === 0}
            onChange={(event) => changeModel(event.target.value)}
          >
            {models.map((model) => (
              <option
                key={model.id}
                value={model.id}
                disabled={!model.available}
              >
                {model.label}
                {model.available ? "" : " — non entraîné"}
              </option>
            ))}
          </select>
          {switching && <span className="model-loading">chargement…</span>}
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      <AnimatePresence mode="wait">
        {phase === "start" && (
          <motion.div
            key="start"
            className="card"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
          >
            <div className="label">Choisis un début</div>
            <div className="openings">
              {openings.map((o) => (
                <button
                  key={o}
                  className={`opening-chip ${o === opening ? "active" : ""}`}
                  onClick={() => setOpening(o)}
                >
                  {o}
                </button>
              ))}
            </div>
            <div className="label">…ou écris le tien</div>
            <textarea
              value={opening}
              onChange={(e) => setOpening(e.target.value)}
              placeholder="Il était une fois…"
            />
            <div className="row">
              <span className="label">Nombre de trous</span>
              <input
                type="range"
                min={2}
                max={6}
                value={nBlanks}
                onChange={(e) => setNBlanks(Number(e.target.value))}
              />
              <strong>{nBlanks}</strong>
            </div>
            <button
              className="btn primary"
              disabled={!opening.trim()}
              onClick={() => start(false)}
            >
              Générer l'histoire
            </button>
          </motion.div>
        )}

        {phase === "loading" && (
          <motion.div
            key="loading"
            className="card"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="label">
              <span className="spinner" />
              le mini-GPT imagine une histoire…
            </div>
          </motion.div>
        )}

        {phase === "fill" && story && (
          <motion.div
            key="fill"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
          >
            <div className="card">
              <StoryText
                template={story.template}
                renderSlot={(i) => {
                  const blank = story.blanks[i];
                  if (answers[blank.index] !== undefined)
                    return (
                      <span className="slot filled">
                        {answers[blank.index]}
                      </span>
                    );
                  if (i === current)
                    return <span className="slot active">?</span>;
                  return (
                    <span className="slot pending">
                      &nbsp;{blank.hint}&nbsp;
                    </span>
                  );
                }}
              />
              <div className="progress">
                <span>
                  Trou {current + 1} / {story.blanks.length}
                </span>
                <div className="progress-dots">
                  {story.blanks.map((b, i) => (
                    <span
                      key={b.index}
                      className={`pdot ${
                        answers[b.index] !== undefined
                          ? "done"
                          : i === current
                            ? "now"
                            : ""
                      }`}
                    />
                  ))}
                </div>
              </div>
            </div>

            <SuggestionList blank={story.blanks[current]} onChoose={choose} />
          </motion.div>
        )}

        {phase === "reveal" && story && (
          <motion.div
            key="reveal"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <ResultCard
              template={story.template}
              answers={answers}
              bertTop={bertTop}
              onReplay={() => start(true)}
              onNew={() => setPhase("start")}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
