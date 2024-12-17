entity ClockGenerator is
port (
  clock : buffer bit;
)
end entity ClockGenerator;

architecture Equations of ClockGenerator is
begin
  clock <= not clock after 5 ns;
end architecture Equations;