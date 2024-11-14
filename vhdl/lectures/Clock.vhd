-- Generatore di clock a 100 MHz
entity ClockGenerator is
  port (clkn : buffer bit);
end ClockGenerator;

architecture equations of ClockGenerator is
begin
  -- Dobbiamo impiegare un buffer, in quanto clkn e' *sia* letto che scritto!
  clkn <= not clkn after 5 ns;
end architecture equations;

