import "@testing-library/jest-dom/vitest";
import { configure } from "@testing-library/react";

// The first test in a file pays the cold-start cost of everything it renders, and with many files running in
// parallel that can exceed testing-library's one-second default for findBy*/waitFor. A longer limit costs
// nothing when things are fast: a wait ends the moment its condition is met.
configure({ asyncUtilTimeout: 5000 });
