# ML Gas Oil Frontend

Professional TypeScript React application with clean architecture.

## Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/       # Reusable UI components
│   │   ├── Header.tsx
│   │   ├── Header.css
│   │   ├── TabNavigation.tsx
│   │   ├── TabNavigation.css
│   │   ├── PrimaryButton.tsx
│   │   └── PrimaryButton.css
│   ├── pages/           # Tab content pages
│   │   ├── OverviewTab.tsx
│   │   ├── OverviewTab.css
│   │   ├── ReportAnalysisTab.tsx
│   │   └── ReportAnalysisTab.css
│   ├── styles/          # Global styles
│   │   ├── variables.css
│   │   └── reset.css
│   ├── types/           # TypeScript definitions
│   │   └── index.ts
│   ├── App.tsx
│   ├── App.css
│   └── index.tsx
├── tsconfig.json
└── package.json
```

## Installation

```bash
cd frontend
npm install
```

## Development

```bash
npm start
```

## Build

```bash
npm run build
```
