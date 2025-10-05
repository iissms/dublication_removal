import { useEffect, useRef } from 'react';
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
  const processedQuestion = nl(mathQuestion);
  const processedAnswer = nl(mathAnswer);

  useEffect(() => {
    if (window.MathJax?.Hub) {
      mathJaxLoaded.current = true;
      window.MathJax.Hub.Queue(['Typeset', window.MathJax.Hub]);
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
          messageStyle: 'none',
          showProcessingMessages: false,
          showMathMenu: false,
          TeX: { extensions: ['mhchem.js'] },
          tex2jax: { inlineMath: [['$', '$'], ['\\(', '\\)']] },
          'HTML-CSS': { linebreaks: { automatic: true } },
          SVG: { linebreaks: { automatic: true } },
        });
        window.MathJax.Hub.Queue(['Typeset', window.MathJax.Hub]);
      };
      document.head.appendChild(script);
    }
  }, []);

  useEffect(() => {
    if (mathJaxLoaded.current && window.MathJax?.Hub) {
      window.MathJax.Hub.Queue(['Typeset', window.MathJax.Hub]);
    }
  }, [processedQuestion, processedAnswer]);

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
      </main>
    </div>
  );
}

export default App;
