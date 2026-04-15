export function decodeMaskBitset(encoded: string, count: number): Uint8Array {
  const raw = Uint8Array.from(atob(encoded), (char) => char.charCodeAt(0));
  const output = new Uint8Array(count);
  for (let i = 0; i < count; i += 1) {
    const byte = raw[i >> 3] ?? 0;
    output[i] = (byte >> (i & 7)) & 1;
  }
  return output;
}
