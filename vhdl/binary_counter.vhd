-- Librerie

entity Count16 is
  port (
    reset : in std_logic;
    clock : in std_logic;
    direction : in std_logic;

    -- Non fosse buffer non potrei metterlo a destra e sinistra degli assegnamenti!
    y : buffer unsigned(3 downto 0);
  )
end entity count16;

architecture Behavioral of Count16 is
begin
  process(reset, clock) is
  begin
    -- Il reset e' *attivo basso*!
    if (reset = '0') then
      y <= (others => '0');
    elsif rising_edge(clock) then
      if (direction = '1') then
        y <= y + 1;
      else
        y <= y - 1;
      end if;
    end if;
  end process;
end architecture Behavioral;