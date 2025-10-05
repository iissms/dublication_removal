import { useCallback, useEffect, useRef, useState } from 'react';
import './App.css';

const mathQuestion = `A series $LCR$ circuit consists of $R = 80\\,\\Omega$, $X_L = 100\\,\\Omega$, and $X_C = 40\\,\\Omega$. The input voltage is $2500 \\cos(100\\pi t)\\,\\text{V}$. The amplitude of current, in the circuit, is ___$A$.`;
const mathAnswer = `Therefore, the amplitude of the current in the circuit is $25\\,\\text{A}$.`;

const nl = (s) => {
  if (typeof s !== 'string') return s;

  let out = s.replace(/\r\n/g, '\n').replace(/\\r\\n/g, '\n');

  out = out
    .replace(/\\\\\[/g, '\\[')
    .replace(/\\\\\]/g, '\\]')
    .replace(/\\\\\(/g, '\\(')
    .replace(/\\\\\)/g, '\\)')
    .replace(/\\\\/g, '\\');

  out = out.replace(/([^\n])\\\[(.*?)\\\]([^\n])/g, (m, left, inner, right) => {
    return `${left}\\(${inner}\\)${right}`;
  });

  return out;
};

function App() {
  const mathJaxLoaded = useRef(false);
  const [mathML, setMathML] = useState({ question: [], answer: [] });
  const processedQuestion = nl(mathQuestion);
  const processedAnswer = nl(mathAnswer);

  const updateMathML = useCallback(() => {
    if (!window.MathJax?.Hub) {
      return;
    }

    const collectMathML = (elementId) => {
      try {
        const jax = window.MathJax.Hub.getAllJax(elementId) || [];
        return jax
          .map((entry) => {
            try {
              return entry?.root?.toMathML?.('') || '';
            } catch (error) {
              console.error('Unable to convert LaTeX to MathML:', error);
              return '';
            }
          })
          .filter(Boolean);
      } catch (error) {
        console.error('Unable to read MathJax output:', error);
        return [];
      }
    };

    setMathML({
      question: collectMathML('math-question'),
      answer: collectMathML('math-answer'),
    });
  }, []);

  useEffect(() => {
    if (window.MathJax?.Hub) {
      mathJaxLoaded.current = true;
      window.MathJax.Hub.Queue(['Typeset', window.MathJax.Hub]);
      window.MathJax.Hub.Queue(updateMathML);
      return;
    }

    if (!mathJaxLoaded.current) {
      const script = document.createElement('script');
      script.src =
        'https://cdn.jsdelivr.net/npm/mathjax@2.7.4/MathJax.js?config=TeX-MML-AM_HTMLorMML';
      script.async = true;
      script.onload = () => {
        mathJaxLoaded.current = true;
        window.MathJax.Hub.Config({
          extensions: ['tex2jax.js', 'toMathML.js'],
          messageStyle: 'none',
          showProcessingMessages: false,
          showMathMenu: false,
          TeX: { extensions: ['mhchem.js'] },
          tex2jax: { inlineMath: [['$', '$'], ['\\(', '\\)']] },
          'HTML-CSS': { linebreaks: { automatic: true } },
          SVG: { linebreaks: { automatic: true } },
        });
        window.MathJax.Hub.Queue(['Typeset', window.MathJax.Hub]);
        window.MathJax.Hub.Queue(updateMathML);
      };
      document.head.appendChild(script);
    }
  }, [updateMathML]);

  useEffect(() => {
    if (!mathJaxLoaded.current || !window.MathJax?.Hub) {
      return undefined;
    }

    let cancelled = false;

    const safeUpdate = () => {
      if (!cancelled) {
        updateMathML();
      }
    };

    window.MathJax.Hub.Queue(['Typeset', window.MathJax.Hub]);
    window.MathJax.Hub.Queue(safeUpdate);

    return () => {
      cancelled = true;
    };
  }, [processedQuestion, processedAnswer, updateMathML]);

  return (
    <div className="App">
      <main className="App-content">
        <h1>Physics Question</h1>
        <p
          id="math-question"
          dangerouslySetInnerHTML={{ __html: processedQuestion }}
        />
        <p
          id="math-answer"
          dangerouslySetInnerHTML={{ __html: processedAnswer }}
        />
        <section className="mathml-output">
          <h2>MathML from MathJax</h2>
          <div className="mathml-section">
            <h3>Question expressions</h3>
            {mathML.question.length > 0 ? (
              mathML.question.map((snippet, index) => (
                <pre className="mathml-code" key={`question-${index}`}>
                  <code>{snippet}</code>
                </pre>
              ))
            ) : (
              <p className="mathml-empty">No MathML captured yet.</p>
            )}
          </div>
          <div className="mathml-section">
            <h3>Answer expressions</h3>
            {mathML.answer.length > 0 ? (
              mathML.answer.map((snippet, index) => (
                <pre className="mathml-code" key={`answer-${index}`}>
                  <code>{snippet}</code>
                </pre>
              ))
            ) : (
              <p className="mathml-empty">No MathML captured yet.</p>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
