const apiFetch = (url: string, options?: RequestInit) =>
  fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  }).then(res => res.json());

export const apiClient = {
  getProject: (id: string) => apiFetch(`/api/projects/${id}`),
  getProjects: () => apiFetch('/api/projects'),
  createProject: (data: any) => apiFetch('/api/projects', { method: 'POST', body: JSON.stringify(data) }),
  startProject: (id: string) => apiFetch(`/api/projects/${id}/start`, { method: 'POST' }),
  pauseProject: (id: string) => apiFetch(`/api/projects/${id}/pause`, { method: 'POST' }),
  resumeProject: (id: string) => apiFetch(`/api/projects/${id}/resume`, { method: 'POST' }),
  getSession: (id: string) => apiFetch(`/api/sessions/${id}`),
  getSessionResults: (id: string) => apiFetch(`/api/sessions/${id}/results`),
  getSystemStatus: () => apiFetch('/api/system/status'),
  healthCheck: () => apiFetch('/api/health'),
  getConcurrentSessions: () => apiFetch('/api/sessions/concurrent'),
  getSessionHistory: () => apiFetch('/api/sessions/history'),
  getActiveProjects: () => apiFetch('/api/projects/active'),
  getProjectSessions: (id: string) => apiFetch(`/api/projects/${id}/sessions`),
  getProjectStatistics: (id: string) => apiFetch(`/api/projects/${id}/statistics`),
  updateSessionProgress: (id: string, data: any) =>
    apiFetch(`/api/sessions/${id}/progress`, { method: 'POST', body: JSON.stringify(data) }),
};

export const queryKeys = {
  project: (id: string) => ['project', id],
  session: (id: string) => ['session', id],
  sessionResults: (id: string) => ['session', id, 'results'],
};

export const api = {
  get: (path: string) => apiFetch(path),
  post: (path: string, data?: any) =>
    apiFetch(path, { method: 'POST', body: data ? JSON.stringify(data) : undefined }),
  put: (path: string, data?: any) =>
    apiFetch(path, { method: 'PUT', body: data ? JSON.stringify(data) : undefined }),
};
