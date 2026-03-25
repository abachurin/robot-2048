# UI Component

React + TypeScript + Vite single-page application for the 2048 game interface.

## Location

`ts-vite-ui/`

## Tech Stack

- React 18 + TypeScript
- Vite (build tool)
- Plotly.js (training charts)
- Axios (HTTP client)

## Key Files

| File | Purpose |
|------|---------|
| `src/App.tsx` | Root component |
| `src/config.ts` | API URL config (localhost:8000 dev, robot2048.com/service prod) |
| `src/api/requests.ts` | API client functions |
| `src/types.ts` | TypeScript type definitions |
| `src/gameLogic.ts` | Client-side 2048 game logic (for human play) |

## Component Structure

```
App
├── Header
│   ├── Logo
│   ├── Login (register/login/logout)
│   ├── SettingsModal (sound, animation, palette)
│   ├── HelpModal
│   └── ContactsModal
├── Main
│   ├── PaneGame (left panel)
│   │   ├── GameBoard (4x4 grid with cells)
│   │   ├── PlayFooter (human play controls)
│   │   ├── WatchFooter (watch agent controls)
│   │   ├── WatchModal (start watch config)
│   │   └── ReplayModal (replay saved game)
│   └── PaneAgent (right panel)
│       ├── LogWindow (streaming training/test logs)
│       ├── Chart (training history plot via Plotly)
│       ├── CurrentJobDescription (running job info)
│       ├── TrainModal (training config)
│       ├── TestModal (test config)
│       └── ManageModal (delete agents/games)
├── Footer
└── StarField (background animation)
```

## State Management

- `src/contexts/UserProvider/` — user auth state (React Context)
- `src/contexts/ModalProvider/` — modal open/close state
- `src/store/gameStore.ts` — game board state (Zustand or similar)
- `src/store/logsStore.ts` — log polling state
- `src/store/modeStore.ts` — play/watch mode

## API Communication

All API calls go through `src/api/requests.ts`. Base URL from `src/config.ts`:
- Development: `http://localhost:8000`
- Production: `https://robot2048.com/service`

## User Features

- **Play**: Human plays 2048 with keyboard/swipe
- **Train**: Configure and launch RL agent training (N, alpha, decay, episodes)
- **Test**: Test agent with lookahead search (depth, width, trigger)
- **Watch**: Watch agent play in real-time (moves streamed from worker)
- **Replay**: Replay saved games move-by-move
- **Charts**: Training progress visualization
- **Logs**: Real-time log streaming from worker
- **Settings**: Sound, animation speed, color palette, tile legends

## Build

```bash
npm run build    # outputs to dist/
npm run dev      # dev server on :5173
```

Static site on DO — built with Node.js buildpack, serves from `dist/`.
