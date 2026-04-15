import React from "react";

type Props = {
  children: React.ReactNode;
};

type State = {
  errorMessage: string | null;
};

export class AppErrorBoundary extends React.Component<Props, State> {
  override state: State = {
    errorMessage: null,
  };

  static override getDerivedStateFromError(error: unknown): State {
    return {
      errorMessage: error instanceof Error ? error.message : "Unknown frontend error.",
    };
  }

  override componentDidCatch(error: unknown, info: React.ErrorInfo) {
    console.error("spark_sam_viewer frontend crash", error, info);
  }

  override render() {
    if (this.state.errorMessage) {
      return (
        <div className="app-crash">
          <div className="app-crash__panel">
            <p className="eyebrow">Frontend Error</p>
            <h1>UI crashed during interaction</h1>
            <p className="hero-copy">{this.state.errorMessage}</p>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
