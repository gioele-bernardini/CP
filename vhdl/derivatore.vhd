-- Libreria

entity Derivatore is
  port (
    s : in std_logic;
    clock : in std_logic;
    reset : in std_logic;

    y : out std_logic;
  )
end entity Derivatore;

architecture Behavioral of Derivatore is
  -- Enumerazione
  type stato is (a, b, c, d);
  signal present_state, next_state : stato;
begin
  -- sequenziale perche' unico processo che risulta in FF
  -- Semplice blocchetto dove viene detto che, se non c'e' reset, lo stato presente si aggiorna (diventa lo stato futuro calcolato prima!)
  seq: process (reset, clock) is
  begin
    if reset = '0' then
      present_state <= a;
    elsif rising_edge(clock) then
      present_state <= next_state;
    end if;
  end process seq;

  futuro: process(present_state, s) is
  begin
    -- Switch statement!
    case present_state is
      when a =>
        if s = '0' then
          next_state <= a;
        else
          next_state <= b;
        end if;
      when b =>
        next_state <= c;
      when c =>
        if s = '0' then
          next_state <= d;
        else
          next_state <= c;
        end if;
      when d =>
        next_state <= a;
    end case;
  end process futuro;

  uscite: process(present_state) is
  begin
    y <= '0'
    if present_state = b or present_state = d then
      y <= '1';
    end if;
  end process uscite;
end architecture Behavioral;

-- Testbench
entity test_seq is
end entity test_seq;

architecture Behavioral of test_seq is
  signal segnali_interni;
begin
  clock: process is
  begin
    clock <= '0';
    wait for 5 ns;
    clock <= '1';
    wait for 5 ns;

  reset: process is
  begin
    reset <= '1';
    wait for 5 ns;
    reset <= '0';
    wait for 50 ns;
    reset <= '1';
    wait;
  end process reset;

  -- Altri segnali
    
        
