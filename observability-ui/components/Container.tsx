interface Props { children: React.ReactNode; maxWidth?: number }

export default function Container({ children, maxWidth = 1200 }: Props) {
  return (
    <div style={{ maxWidth, margin: "0 auto", padding: "0 32px" }}>
      {children}
    </div>
  );
}
