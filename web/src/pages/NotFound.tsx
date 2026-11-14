import { PageHead } from '../components/chrome';
import { Container } from '../components/chrome';
import { Link, useTitle } from '../lib/router';

export function NotFound() {
  useTitle('Page not found');
  return (
    <>
      <PageHead eyebrow="404" title="Nothing on this slice.">
        The page you asked for does not exist.
      </PageHead>
      <Container className="pb-32">
        <Link to="/" className="mr-6">Go to the home page</Link>
        <Link to="/analyse">Analyse a scan</Link>
      </Container>
    </>
  );
}
