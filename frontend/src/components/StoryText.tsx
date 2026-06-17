import { ReactNode } from "react";

export type Segment = { text: string } | { blank: number };

export function parseTemplate(template: string): Segment[] {
  return template.split(/\{\{(\d+)\}\}/).map((part, index) =>
    index % 2 === 0 ? { text: part } : { blank: Number(part) }
  );
}

interface Props {
  template: string;
  renderSlot: (blankIndex: number) => ReactNode;
}

export default function StoryText({ template, renderSlot }: Props) {
  return (
    <p className="story">
      {parseTemplate(template).map((segment, i) =>
        "text" in segment ? (
          <span key={i}>{segment.text}</span>
        ) : (
          <span key={i}>{renderSlot(segment.blank)}</span>
        )
      )}
    </p>
  );
}
