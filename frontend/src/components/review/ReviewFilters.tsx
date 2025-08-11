export interface ReviewFiltersProps {
  onFilterChange?: (filters: any) => void;
}

export function ReviewFilters({ onFilterChange }: ReviewFiltersProps) {
  return (
    <div>
      <p className="text-sm text-muted-foreground">Review filters component placeholder</p>
      <button
        className="mt-2 px-4 py-2 bg-gray-200 rounded"
        onClick={() => onFilterChange?.({})}
      >
        Apply Filters
      </button>
    </div>
  );
}
