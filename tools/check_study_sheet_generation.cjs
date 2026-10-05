// Synthetic prompt/response QA. Reads the configured model key without logging
// or saving it. No student records or Firestore connections are used.
const fs = require('fs'), path = require('path');
const root = path.resolve(__dirname, '..');
const output = path.resolve(process.argv[2]);
const env = Object.fromEntries(fs.readFileSync(path.join(root, '.env'), 'utf8').split(/\r?\n/).flatMap(line => {
  const m = line.match(/^\s*([A-Z_]+)\s*=\s*(.*?)\s*$/); return m ? [[m[1], m[2].replace(/^['"]|['"]$/g, '')]] : [];
}));
const system = fs.readFileSync(path.join(root, 'services/studysheet_simple.py'), 'utf8').match(/SYSTEM = """([\s\S]*?)"""/)[1];
const planSystem = fs.readFileSync(path.join(root, 'services/studysheet_simple.py'), 'utf8').match(/PLAN_SYSTEM = """([\s\S]*?)"""/)[1];
const reviewSystem = fs.readFileSync(path.join(root, 'services/studysheet_simple.py'), 'utf8').match(/SOURCE_REVIEW_SYSTEM = """([\s\S]*?)"""/)[1];
const common = { language: 'english', topicHint: 'Study guide', conversation: [], studentSignals: {}, quizzes: [], previousSheetsForRevisions: [], fileTopics: {}, priorities: [], sources: [], materials: [] };
const cases = [
  { name: 'focused-statistics', payload: { ...common,
    currentRequest: 'Make a concise, hand-copyable study sheet ONLY about mean versus median and outliers. My teacher said outliers are very important. Review my relevant quiz mistake. No practice questions. Do not include sociology.',
    conversation: [{ role: 'user', content: 'I mix up mean and median. I want to understand why an outlier changes the mean.' }],
    priorities: [{ sourceId: 'P1', quote: 'My teacher said outliers are very important.' }],
    sources: [{ id: 'D1', kind: 'document', label: 'Statistics notes' }, { id: 'P1', kind: 'conversation', label: 'Your study request' },
      { id: 'Q1', kind: 'quiz', label: 'Statistics quiz', answered: 1, total: 2 }, { id: 'Q2', kind: 'quiz', label: 'Unrelated sociology quiz', answered: 1, total: 1 }],
    materials: [{ sourceId: 'D1', text: 'Mean = sum divided by the number of observations. Median = middle observation after sorting. Outliers pull the mean toward their value; the median is generally less affected. Example: 2, 3, 4 has mean and median 3. For 2, 3, 100 the median is 3 and the mean is 35.' }],
    quizzes: [{ sourceId: 'Q1', answered: 1, total: 2, questions: [{ question: 'Which measure is generally less affected by an outlier?', topic: 'Mean vs median', options: ['Mean', 'Median'], correctAnswer: 'Median', latestAttempt: { selectedOption: 'Mean', isCorrect: false } }, { question: 'Calculate a mean.', latestAttempt: null }] },
      { sourceId: 'Q2', answered: 1, total: 1, questions: [{ topic: 'Sociology', question: 'Define social roles.', latestAttempt: { isCorrect: false } }] }] } },
  { name: 'french-history', payload: { ...common, language: 'french',
    currentRequest: 'Refais-la en français, plus détaillée sur le mécanisme uniquement, sans médicaments et sans questions.',
    conversation: [{ role: 'user', content: 'I need a study sheet about glomerulonephritis. My teacher emphasized how glomerular injury affects filtration.' }],
    priorities: [{ sourceId: 'P1', quote: 'My teacher emphasized how glomerular injury affects filtration.' }],
    sources: [{ id: 'P1', kind: 'conversation', label: 'Votre indication' }, { id: 'D1', kind: 'document', label: 'Notes rénales' }],
    materials: [{ sourceId: 'D1', text: 'Glomerulonephritis is inflammation and injury of the glomeruli, the filtering units of the kidney. The injury changes normal filtration. It can be acute, rapidly progressive, or chronic. This excerpt does not specify laboratory ranges or drug treatments.' }],
    previousSheetsForRevisions: [{ title: 'Glomerulonephritis', content: 'A broad guide about glomerular injury, related syndromes, filtration markers and medications.' }] } },
];
async function main() {
  if (!env.OPENAI_API_KEY) throw new Error('No configured model key');
  fs.mkdirSync(output, { recursive: true });
  for (const test of cases.filter(c => !process.argv[3] || c.name === process.argv[3])) {
    const planResponse = await fetch('https://api.openai.com/v1/chat/completions', { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${env.OPENAI_API_KEY}` },
      body: JSON.stringify({ model: env.OPENAI_STUDY_SHEET_MODEL || 'gpt-4.1-mini', temperature: 0, max_completion_tokens: 3000, response_format: { type: 'json_object' },
        messages: [{ role: 'system', content: planSystem }, { role: 'user', content: JSON.stringify(test.payload) }] }) });
    if (!planResponse.ok) throw new Error('Plan check HTTP ' + planResponse.status);
    const planned = await planResponse.json();
    test.payload.resolvedPlan = JSON.parse(planned.choices[0].message.content);
    test.payload.language = test.payload.resolvedPlan.language || test.payload.language;
    const selected = new Set([...test.payload.resolvedPlan.priorityIds, ...test.payload.resolvedPlan.quizIds]);
    test.payload.sources = test.payload.sources.filter(s => !['quiz', 'conversation'].includes(s.kind) || selected.has(s.id));
    test.payload.quizzes = test.payload.quizzes.filter(q => selected.has(q.sourceId));
    test.payload.priorities = test.payload.priorities.filter(p => selected.has(p.sourceId));
    test.payload.focusPrepared = selected.size > 0;
    test.payload.layoutDensity = /concise|compact|hand.copy|brief|one.page|succinct|bref|brève|recopier/i.test(test.payload.resolvedPlan.depth) ? 'compact' : 'comfortable';
    const writing = { ...test.payload };
    if (writing.focusPrepared) {
      writing.sources = writing.sources.filter(s => ['document', 'notes'].includes(s.kind));
      delete writing.priorities; delete writing.quizzes;
      writing.preparedFocus = test.payload.resolvedPlan.practiceReviews;
      writing.resolvedPlan = Object.fromEntries(Object.entries(writing.resolvedPlan).filter(([k]) => !['priorityIds', 'quizIds', 'practiceReviews'].includes(k)));
    }
    const response = await fetch('https://api.openai.com/v1/chat/completions', { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${env.OPENAI_API_KEY}` },
      body: JSON.stringify({ model: env.OPENAI_STUDY_SHEET_MODEL || 'gpt-4.1-mini', temperature: .3, max_completion_tokens: 12000, response_format: { type: 'json_object' },
        messages: [{ role: 'system', content: system }, { role: 'user', content: JSON.stringify(writing) }] }) });
    if (!response.ok) throw new Error('Generation check HTTP ' + response.status);
    const result = await response.json();
    if (result.choices[0].finish_reason !== 'stop') throw new Error('Incomplete check response');
    const blocks = JSON.parse(result.choices[0].message.content).records.filter(r => r.type === 'section')
      .flatMap(r => r.blocks.map(b => ({ sectionTitle: r.title, ...b }))).map((b, i) => ({ id: `B${i}`, ...b }));
    const reviewedResponse = await fetch('https://api.openai.com/v1/chat/completions', { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${env.OPENAI_API_KEY}` },
      body: JSON.stringify({ model: env.OPENAI_STUDY_SHEET_REVIEW_MODEL || 'gpt-4.1', temperature: 0, max_completion_tokens: 6000, response_format: { type: 'json_object' },
        messages: [{ role: 'system', content: reviewSystem }, { role: 'user', content: JSON.stringify({ materials: test.payload.materials, blocks }) }] }) });
    if (!reviewedResponse.ok) throw new Error('Attribution check HTTP ' + reviewedResponse.status);
    const reviewed = await reviewedResponse.json();
    if (reviewed.choices[0].finish_reason !== 'stop') throw new Error('Incomplete attribution response');
    fs.writeFileSync(path.join(output, test.name + '.ndjson'), result.choices[0].message.content);
    fs.writeFileSync(path.join(output, test.name + '.sources.json'), JSON.stringify(test.payload.sources));
    fs.writeFileSync(path.join(output, test.name + '.plan.json'), JSON.stringify(test.payload.resolvedPlan));
    fs.writeFileSync(path.join(output, test.name + '.payload.json'), JSON.stringify(test.payload));
    fs.writeFileSync(path.join(output, test.name + '.review.json'), reviewed.choices[0].message.content);
    console.log(JSON.stringify({ name: test.name, model: result.model, tokens: result.usage.total_tokens }));
  }
}
main().catch(error => { console.error(error.message); process.exitCode = 1; });
